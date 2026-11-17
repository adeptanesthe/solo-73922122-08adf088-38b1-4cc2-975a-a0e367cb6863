"""送检选样：按随机森林叶节点覆盖收益挑选待检配方。

收益语义（严格按题目）：
- 模型输入沿用固定随机森林的 float32（克数 / 1000）；
- 每个叶节点的收益 = 1 / 该叶训练样本数（训练样本越少收益越高）；
- 已检配方命中的叶节点不再计入收益；
- 同一叶被多条待检配方命中只计一次（叶级去重）；
- 在 1..名额 条待检配方中找收益最高的组合；同分时送检数最少；
- 没有任何增量收益则不推荐送检；穷举未能在预算内证明最优则保持待定。

森林规模固定为 5 棵树、每棵至多 8 个叶（共至多 40 叶），待检至多 60 条、
名额至多 5，故用整数位掩码 + 分支定界穷举全部可行组合，可严格证明最优；
若超出时间/节点预算（防御性兜底，正常参数下远不会触及）则返回 pending。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction
from math import gcd as _gcd

from .model import ForestSpec, Leaf, N_FEATURES, TOTAL_GRAMS, cutoff_left

# 状态取值
SEL_OK = "ok"                    # 已穷举证明最优的送检组合（含“不推荐送检”）
SEL_INVALID = "invalid"          # 输入不合法
SEL_PENDING = "pending"          # 预算内未能证明最优，保持待定

# 防御性预算：当前固定森林（≤40 叶）× 上限输入（60 选 5）下实测远用不到
TIME_BUDGET_SECONDS = 5.0
NODE_BUDGET = 20_000_000

# 输入数量边界
MIN_TESTED, MAX_TESTED = 0, 20
MIN_CANDIDATES, MAX_CANDIDATES = 10, 60
MIN_SLOTS, MAX_SLOTS = 1, 5


@dataclass(frozen=True)
class LeafInfo:
    tree: int          # 树序号
    node: int          # 树内节点 id
    n_train: int       # 该叶训练样本数
    weight: Fraction   # 1 / n_train


class LeafRegistry:
    """全森林叶节点的统一编号（位掩码的 bit 位）与收益表。

    为避免穷举中反复做 Fraction 的 gcd 运算，把所有叶收益 1/n_train 放大
    LCM(n_train) 倍存为精确整数；对外仍可换算回 Fraction，结果完全一致。
    """

    def __init__(self, forest: ForestSpec):
        self.leaves: list[LeafInfo] = []
        self.index: dict[tuple[int, int], int] = {}
        for ti, tree in enumerate(forest.trees):
            for nid, node in enumerate(tree.nodes):
                if isinstance(node, Leaf):
                    pos = len(self.leaves)
                    assert node.n_train > 0
                    self.leaves.append(
                        LeafInfo(ti, nid, node.n_train, Fraction(1, node.n_train))
                    )
                    self.index[(ti, nid)] = pos
        self.n_leaves = len(self.leaves)
        scale = 1
        for info in self.leaves:
            scale = scale * info.n_train // _gcd(scale, info.n_train)
        self.scale = scale
        # 每个 bit 放大后的整数收益
        self._bit_int = [scale // info.n_train for info in self.leaves]
        self._mask_int: dict[int, int] = {0: 0}

    def int_benefit(self, mask: int) -> int:
        """位掩码的放大整数收益（memoized；穷举中掩码大量复用）。"""
        cached = self._mask_int.get(mask)
        if cached is not None:
            return cached
        low = mask & -mask
        val = self._bit_int[low.bit_length() - 1] + self.int_benefit(mask ^ low)
        if len(self._mask_int) < 200_000:  # 防御性上限，避免极端输入下内存膨胀
            self._mask_int[mask] = val
        return val

    def mask_benefit(self, mask: int) -> Fraction:
        return Fraction(self.int_benefit(mask), self.scale)


def route_formula(forest: ForestSpec, grams: list[int], registry: "LeafRegistry | None" = None) -> int:
    """按 float32 输入语义把一条配方路由到 5 个叶，返回叶位掩码（每树恰 1 bit）。

    走树规则与 model.forest_proba_rational 一致：用 cutoff_left 把
    “float32(g/1000) <= 阈值”精确翻译成整数克数比较。
    """
    reg = registry or get_registry(forest)
    mask = 0
    for ti, tree in enumerate(forest.trees):
        nid = tree.root
        while True:
            node = tree.nodes[nid]
            if isinstance(node, Leaf):
                break
            if grams[node.feature] <= cutoff_left(node.threshold):
                nid = node.left
            else:
                nid = node.right
        mask |= 1 << reg.index[(ti, nid)]
    return mask


# 进程内森林固定为启动时训练的同一实例，缓存其注册表即可；
# 自测训练新森林时通过 reset 清理，避免跨实例的陈旧缓存。
_REGISTRY_CACHE: dict[int, LeafRegistry] = {}


def get_registry(forest: ForestSpec) -> LeafRegistry:
    reg = _REGISTRY_CACHE.get(id(forest))
    if reg is None:
        reg = LeafRegistry(forest)
        _REGISTRY_CACHE.clear()
        _REGISTRY_CACHE[id(forest)] = reg
    return reg


def reset_leaf_cache() -> None:
    _REGISTRY_CACHE.clear()


def _validate_formula(label: str, row, seen: set[tuple[int, ...]]) -> tuple[list[int] | None, str | None]:
    """校验单条配方；seen 为同组去重集合（已检/待检各自独立去重）。"""
    if not isinstance(row, (list, tuple)) or len(row) != N_FEATURES:
        return None, f"{label}：每条配方必须是 {N_FEATURES} 个整数"
    out = []
    for x in row:
        if isinstance(x, bool) or not isinstance(x, int):
            return None, f"{label}：用量必须是整数克数"
        if not (0 <= x <= TOTAL_GRAMS):
            return None, f"{label}：每种原料用量须在 0~{TOTAL_GRAMS} 克之间"
        out.append(x)
    if sum(out) != TOTAL_GRAMS:
        return None, f"{label}：配方总重须为 {TOTAL_GRAMS} 克（当前 {sum(out)} 克）"
    key = tuple(out)
    if key in seen:
        return None, f"{label}：存在重复配方 {out}，请去重后再提交"
    seen.add(key)
    return out, None


@dataclass
class SelectionInput:
    tested: list[list[int]]       # 已检配方 0~20 条
    candidates: list[list[int]]   # 待检配方 10~60 条
    slots: int                    # 送检名额 1~5


def validate_selection(inp: SelectionInput) -> list[str]:
    errs: list[str] = []
    n_tested = len(inp.tested) if isinstance(inp.tested, list) else -1
    n_cand = len(inp.candidates) if isinstance(inp.candidates, list) else -1
    if not MIN_TESTED <= n_tested <= MAX_TESTED:
        errs.append(f"已检配方数量须在 {MIN_TESTED}~{MAX_TESTED} 条之间（当前 {n_tested}）")
    if not MIN_CANDIDATES <= n_cand <= MAX_CANDIDATES:
        errs.append(f"待检配方数量须在 {MIN_CANDIDATES}~{MAX_CANDIDATES} 条之间（当前 {n_cand}）")
    if isinstance(inp.slots, bool) or not isinstance(inp.slots, int) or not (MIN_SLOTS <= inp.slots <= MAX_SLOTS):
        errs.append(f"送检名额须为 {MIN_SLOTS}~{MAX_SLOTS} 之间的整数")
    if errs:
        return errs
    # 两组各自独立去重；同一配方允许同时出现在已检与待检中（其增量掩码自然为 0）
    for label, rows in (("已检配方", inp.tested), ("待检配方", inp.candidates)):
        seen: set[tuple[int, ...]] = set()
        for i, row in enumerate(rows):
            _, err = _validate_formula(f"{label}#{i + 1}", row, seen)
            if err:
                errs.append(err)
    return errs


@dataclass
class SelectionResult:
    status: str
    reason: str = ""
    selected: list[list[int]] | None = None          # 选中的待检配方（按提交顺序）
    selected_indices: list[int] | None = None        # 选中配方在待检列表中的下标（0 起）
    gain_mask: int = 0                                # 增量覆盖叶位掩码
    gain_leaves: list[dict] | None = None             # 增量叶明细
    benefit: float | None = None                      # 增量收益（float 展示）
    benefit_exact: str | None = None                  # 增量收益（精确分数字符串）
    tested_leaf_count: int = 0
    candidate_leaf_count: int = 0                     # 待检配方整体可新增的叶数
    exhaustive: bool = False
    nodes_visited: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "selected": self.selected,
            "selected_indices": self.selected_indices,
            "gain_leaves": self.gain_leaves,
            "benefit": self.benefit,
            "benefit_exact": self.benefit_exact,
            "tested_leaf_count": self.tested_leaf_count,
            "candidate_leaf_count": self.candidate_leaf_count,
            "exhaustive": self.exhaustive,
            "nodes_visited": self.nodes_visited,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


def _leaf_dicts(registry: LeafRegistry, mask: int) -> list[dict]:
    out = []
    for pos, info in enumerate(registry.leaves):
        if mask & (1 << pos):
            out.append({
                "tree": info.tree,
                "node": info.node,
                "n_train": info.n_train,
                "weight": float(info.weight),
            })
    return out


def select(forest: ForestSpec, inp: SelectionInput) -> SelectionResult:
    """分支定界穷举，返回已证明最优的组合；预算耗尽则 pending。"""
    t0 = time.perf_counter()
    errs = validate_selection(inp)
    if errs:
        return SelectionResult(status=SEL_INVALID, reason="；".join(errs))

    registry = get_registry(forest)
    # 已检配方覆盖的叶（并集；重复命中自然去重）
    tested_mask = 0
    for row in inp.tested:
        tested_mask |= route_formula(forest, row, registry)

    # 每条待检配方的“增量掩码”：扣除已检叶后，它能新覆盖的叶
    cand_masks = [route_formula(forest, row, registry) & ~tested_mask for row in inp.candidates]
    candidate_union = 0
    for m in cand_masks:
        candidate_union |= m
    candidate_leaf_count = candidate_union.bit_count()

    elapsed_invalid = (time.perf_counter() - t0) * 1000
    if candidate_union == 0:
        # 待检配方命中的叶全部已被已检配方覆盖 → 无增量收益，不推荐送检
        return SelectionResult(
            status=SEL_OK,
            reason="所有待检配方命中的叶节点均已被已检配方覆盖，没有增量收益，不推荐送检。",
            selected=[],
            selected_indices=[],
            gain_mask=0,
            gain_leaves=[],
            benefit=0.0,
            benefit_exact="0",
            tested_leaf_count=tested_mask.bit_count(),
            candidate_leaf_count=0,
            exhaustive=True,
            elapsed_ms=elapsed_invalid,
        )

    # 去掉增量为 0 的待检（选它们只会平白增加送检数，永不会是最优组合的成员）
    active = [(i, m) for i, m in enumerate(cand_masks) if m]
    # 按掩码收益降序，提升分支定界剪枝效率
    active.sort(key=lambda im: registry.int_benefit(im[1]), reverse=True)
    k = min(inp.slots, len(active))

    # 槽位感知上界（有效上界，可安全剪枝）：任一 r 元组合的新增并集收益，
    # 不超过其成员各自新增收益之和，故不超过剩余候选中 r 个最大单候选增量之和。
    bound_cache: dict[tuple[int, int, int], int] = {}

    def slot_bound(start: int, union: int, r: int) -> int:
        key = (start, union, r)
        cached = bound_cache.get(key)
        if cached is not None:
            return cached
        vals = sorted(
            (registry.int_benefit(active[j][1] & ~union) for j in range(start, len(active))),
            reverse=True,
        )
        total = sum(vals[:r])
        if len(bound_cache) < 50_000:
            bound_cache[key] = total
        return total

    deadline = t0 + TIME_BUDGET_SECONDS
    nodes = 0

    # 贪心种子：让分支定界一开始就有像样的 incumbent，最大化剪枝
    greedy_mask, greedy_chosen = 0, []
    remaining_pool = list(range(len(active)))
    for _ in range(k):
        best_j, best_add = None, 0
        for j in remaining_pool:
            add = active[j][1] & ~greedy_mask
            v = registry.int_benefit(add)
            if v > best_add:
                best_j, best_add = j, v
        if best_j is None or best_add == 0:
            break
        remaining_pool.remove(best_j)
        greedy_chosen.append(best_j)
        greedy_mask |= active[best_j][1]
    greedy_chosen.sort()
    best = {
        "benefit": registry.int_benefit(greedy_mask),
        "count": len(greedy_chosen),
        "indices": greedy_chosen,
    }

    def dfs(start: int, chosen: list[int], union: int, remaining: int) -> bool:
        """返回 False 表示预算耗尽（待定）。不变式：chosen 为 active 下标列表。"""
        nonlocal nodes
        nodes += 1
        if (nodes & 0xFF) == 0 and (nodes > NODE_BUDGET or time.perf_counter() > deadline):
            return False
        cur_ben = registry.int_benefit(union)
        # 同分送检数最少：仅当收益严格更大，或收益相同且条数更少才替换
        if cur_ben > best["benefit"] or (cur_ben == best["benefit"] and len(chosen) < best["count"]):
            best["benefit"] = cur_ben
            best["count"] = len(chosen)
            best["indices"] = list(chosen)
        if remaining == 0 or start >= len(active):
            return True
        # 槽位感知上界：剩余名额内最好也超不过 incumbent，则剪枝
        if cur_ben + slot_bound(start, union, remaining) < best["benefit"]:
            return True
        # 组合枚举：active 已按收益降序；依次尝试把 active[j] 纳入组合
        for j in range(start, len(active)):
            new_union = union | active[j][1]
            # 该候选不带来新叶：选它只会增加送检数，跳过（同分最少条数要求）
            if new_union == union:
                continue
            chosen.append(j)
            if not dfs(j + 1, chosen, new_union, remaining - 1):
                return False
            chosen.pop()
        return True

    finished = dfs(0, [], 0, k)
    elapsed = (time.perf_counter() - t0) * 1000
    if not finished:
        return SelectionResult(
            status=SEL_PENDING,
            reason=(
                f"分支定界穷举在预算（{TIME_BUDGET_SECONDS:.0f} 秒 / {NODE_BUDGET:,} 节点）内"
                "未能证明最优，按规则保持待定，不给出送检建议。"
            ),
            tested_leaf_count=tested_mask.bit_count(),
            candidate_leaf_count=candidate_leaf_count,
            exhaustive=False,
            nodes_visited=nodes,
            elapsed_ms=elapsed,
        )

    chosen_active = best["indices"]
    chosen_orig = sorted(active[j][0] for j in chosen_active)
    gain_mask = 0
    for j in chosen_active:
        gain_mask |= active[j][1]
    gain_benefit = registry.mask_benefit(gain_mask)
    if gain_benefit <= 0:  # 理论上 candidate_union 非 0 时不会发生，防御性处理
        return SelectionResult(
            status=SEL_OK,
            reason="没有增量收益，不推荐送检。",
            selected=[],
            selected_indices=[],
            gain_mask=0,
            gain_leaves=[],
            benefit=0.0,
            benefit_exact="0",
            tested_leaf_count=tested_mask.bit_count(),
            candidate_leaf_count=candidate_leaf_count,
            exhaustive=True,
            nodes_visited=nodes,
            elapsed_ms=elapsed,
        )
    return SelectionResult(
        status=SEL_OK,
        reason="已穷举全部可行组合并证明最优：在名额内不存在收益更高的组合；同收益下该组合送检条数最少。",
        selected=[list(inp.candidates[i]) for i in chosen_orig],
        selected_indices=chosen_orig,
        gain_mask=gain_mask,
        gain_leaves=_leaf_dicts(registry, gain_mask),
        benefit=float(gain_benefit),
        benefit_exact=str(gain_benefit),
        tested_leaf_count=tested_mask.bit_count(),
        candidate_leaf_count=candidate_leaf_count,
        exhaustive=True,
        nodes_visited=nodes,
        elapsed_ms=elapsed,
    )
