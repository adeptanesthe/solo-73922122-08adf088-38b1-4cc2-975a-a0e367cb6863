"""关键功能自测（开发环境运行，不进镜像）：

  python -m tests.selftest

覆盖：
 1. 模型规格：5 棵树、深度 ≤ 3、同种子重训结果一致；
 2. 阈值语义：cutoff_left 对全部树阈值 × 全部克数逐一与 float32 比较核对；
    随机配方上纯 Python 走树概率与 sklearn predict_proba 完全一致；
 3. 可行样例：求解 ok、概率复算达标、改动量一致，并用 z3 反证 + sklearn 暴力
    枚举双重核对最优性结论；
 4. 无解样例：z3 判 unsat，并对整个可行域用 sklearn 暴力枚举复核确无合格配方；
 5. 字典序：改 2 种（量大）与改 3 种（量小）并存时，必须返回改 2 种；
 6. 边界：原配方已合格（0 改动）、全锁定合格/不合格、各类非法输入；
 7. 求解失败路径：异常注入返回 solver_failure 而非无解；
 8. API：/、/api/meta、/api/samples、/api/solve 各状态分支；页面含迟到结果丢弃逻辑；
 9. 送检选样：叶路由与 sklearn apply 一致；收益 = 1/n_train；已检叶不计入、
    同叶去重；穷举最优与同分最少条数（暴力枚举对照）；无增量不推荐；
    预算耗尽保持 pending；非法输入；/api/select 与 /select 分支。
"""
from __future__ import annotations

import itertools
import sys
from functools import reduce as _reduce

import numpy as np
import z3


def functools_reduce_or(masks):
    return _reduce(lambda a, b: a | b, masks, 0)

from app.model import (
    MAX_DEPTH,
    N_TREES,
    TOTAL_GRAMS,
    cutoff_left,
    extract_forest,
    forest_proba_rational,
    is_qualified,
    predict_proba_qualified,
    train_model,
)
from app.solver import (
    INFEASIBLE,
    INVALID,
    OK,
    SOLVER_FAILURE,
    SolveInput,
    _build_optimize,
    solve,
)

PASS, FAIL = "PASS", "FAIL"
failures = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def main():
    print("== 1. 模型规格 ==")
    clf = train_model()
    forest = extract_forest(clf)
    check("树数量为 5", len(clf.estimators_) == N_TREES == 5)
    depths = [e.tree_.max_depth for e in clf.estimators_]
    check("树深均 ≤ 3", all(d <= MAX_DEPTH for d in depths), f"depths={depths}")
    clf2 = train_model()
    same = np.allclose(
        clf.predict_proba(np.zeros((1, 4), dtype=np.float32) + 0.25),
        clf2.predict_proba(np.zeros((1, 4), dtype=np.float32) + 0.25),
    )
    check("同种子重训结果一致", same)

    print("== 2. 阈值语义 ==")
    # cutoff_left：对森林中全部阈值 × 全部整数克数，逐一与 float32 比较核对
    thr_ok = True
    for tree in forest.trees:
        for node in tree.nodes:
            if node.__class__.__name__ == "Leaf":
                continue
            k = cutoff_left(node.threshold)
            for g in range(TOTAL_GRAMS + 1):
                expect = bool(np.float32(g / TOTAL_GRAMS) <= node.threshold)
                if (g <= k) != expect:
                    thr_ok = False
                    break
            if not thr_ok:
                break
    check("cutoff_left 与 float32 比较逐点一致", thr_ok)

    rng = np.random.default_rng(99)
    proba_ok = True
    for _ in range(500):
        g = rng.multinomial(TOTAL_GRAMS, [0.3, 0.3, 0.2, 0.2]).tolist()
        p_sk = predict_proba_qualified(clf, g)
        p_py = float(forest_proba_rational(forest, g))
        if abs(p_sk - p_py) > 1e-12:
            proba_ok = False
            break
    check("500 个随机配方概率与 sklearn 完全一致", proba_ok)

    print("== 3. 可行样例 ==")
    from app.samples import build_samples

    samples = build_samples(forest, clf)
    fs = samples["feasible"]
    inp = SolveInput(fs["original"], fs["lower"], fs["upper"], fs["locked"])
    res = solve(forest, clf, inp)
    check("可行样例求解为 ok", res.status == OK, res.reason)
    check("建议概率 ≥ 0.7（原分类器复算）", res.probability is not None and is_qualified(res.probability),
          f"p={res.probability:.6f}")
    sug = res.suggestion
    check("建议满足总重/上下限/锁定",
          sum(sug) == TOTAL_GRAMS
          and all(fs["lower"][i] <= sug[i] <= fs["upper"][i] for i in range(4))
          and all(sug[i] == fs["original"][i] for i in range(4) if fs["locked"][i]))
    k, m = res.changed_count, res.total_change
    real_k = sum(1 for i in range(4) if sug[i] != fs["original"][i])
    real_m = sum(abs(sug[i] - fs["original"][i]) for i in range(4))
    check("报告的改动种数/克数与建议一致", (k, m) == (real_k, real_m), f"k={k}, m={m}")

    # 3a. z3 反证最优性
    opt, g, num_changed, total_change = _build_optimize(forest, inp)
    opt.add(num_changed <= k - 1)
    check("z3 反证：更少种数无解", opt.check() == z3.unsat)
    opt2, g2, num_changed2, total_change2 = _build_optimize(forest, inp)
    opt2.add(num_changed2 == k, total_change2 <= m - 1)
    check("z3 反证：同种数更小改动量无解", opt2.check() == z3.unsat)

    # 3b. sklearn 暴力枚举复核最优性（改动 ≤2 种且更优者不存在）
    def brute_best(inp_d, max_kinds):
        orig, lo, hi, lk = inp_d["original"], inp_d["lower"], inp_d["upper"], inp_d["locked"]
        free = [i for i in range(4) if not lk[i]]
        best = None  # (kinds, delta)
        for n_k in range(0, max_kinds + 1):
            for S in itertools.combinations(free, n_k):
                fixed_idx = [i for i in range(4) if i not in S]
                fixed_sum = sum(orig[i] for i in fixed_idx)
                if len(S) == 0:
                    cand = [tuple(orig)]
                elif len(S) == 1:
                    i = S[0]
                    gi = TOTAL_GRAMS - fixed_sum
                    cand = [] if not (lo[i] <= gi <= hi[i]) else [tuple(orig[:i]) + (gi,) + tuple(orig[i+1:])]
                else:  # len(S) == 2：和约束下第二变量被唯一确定
                    i, j = S
                    cand = []
                    for gi in range(lo[i], hi[i] + 1):
                        gj = TOTAL_GRAMS - fixed_sum - gi
                        if lo[j] <= gj <= hi[j]:
                            row = list(orig); row[i], row[j] = gi, gj
                            cand.append(tuple(row))
                if not cand:
                    continue
                arr = np.array(cand, dtype=np.float32) / TOTAL_GRAMS
                probs = clf.predict_proba(arr)[:, 1]
                for row, p in zip(cand, probs):
                    if p >= 0.7 - 1e-9:
                        d = sum(abs(row[t] - orig[t]) for t in range(4))
                        cur = (n_k, d)
                        if best is None or cur < best:
                            best = cur
        return best

    bb = brute_best(fs, 2)
    check("sklearn 暴力枚举复核最优性", bb == (k, m), f"brute={bb}, solver=({k},{m})")

    print("== 4. 无解样例 ==")
    isf = samples["infeasible"]
    res_i = solve(forest, clf, SolveInput(isf["original"], isf["lower"], isf["upper"], isf["locked"]))
    check("无解样例求解为 infeasible", res_i.status == INFEASIBLE, res_i.reason)
    # 对整个可行域用 sklearn 暴力枚举复核“确实无解”
    lo, hi = isf["lower"], isf["upper"]
    combos = []
    for g0 in range(lo[0], hi[0] + 1):
        for g1 in range(lo[1], hi[1] + 1):
            for g2 in range(lo[2], hi[2] + 1):
                g3 = TOTAL_GRAMS - g0 - g1 - g2
                if lo[3] <= g3 <= hi[3]:
                    combos.append((g0, g1, g2, g3))
    found_q = False
    for s in range(0, len(combos), 2_000_000):
        arr = np.array(combos[s:s + 2_000_000], dtype=np.float32) / TOTAL_GRAMS
        if (clf.predict_proba(arr)[:, 1] >= 0.7 - 1e-9).any():
            found_q = True
            break
    check(f"sklearn 全可行域枚举（{len(combos)} 点）确无合格配方", not found_q)

    print("== 5. 字典序优先级 ==")
    lex_orig = [244, 294, 218, 244]  # 改 2 种最少 218 克；改 3 种可 214 克
    res_lex = solve(forest, clf, SolveInput(lex_orig, [0]*4, [1000]*4, [False]*4))
    check("先最少种数再最小克数", res_lex.status == OK and res_lex.changed_count == 2
          and res_lex.total_change == 218,
          f"got kinds={res_lex.changed_count}, delta={res_lex.total_change}")

    print("== 6. 边界与非法输入 ==")
    r_ok0 = solve(forest, clf, SolveInput([700, 100, 100, 100], [0]*4, [1000]*4, [False]*4))
    check("原配方已合格 → 0 改动", r_ok0.status == OK and r_ok0.changed_count == 0
          and r_ok0.total_change == 0 and r_ok0.suggestion == [700, 100, 100, 100])
    r_lock_bad = solve(forest, clf, SolveInput([150, 130, 620, 100], [0]*4, [1000]*4, [True]*4))
    check("全锁定且不合格 → infeasible", r_lock_bad.status == INFEASIBLE)
    r_lock_ok = solve(forest, clf, SolveInput([700, 100, 100, 100], [0]*4, [1000]*4, [True]*4))
    check("全锁定且合格 → ok 0 改动", r_lock_ok.status == OK and r_lock_ok.changed_count == 0)
    bad_cases = [
        SolveInput([250, 250, 350, 149], [0]*4, [1000]*4, [False]*4),   # 总和 999
        SolveInput([250, 250, 350, 150], [0, 0, 500, 0], [1000, 1000, 400, 1000], [False]*4),  # lo>hi
        SolveInput([250, 250, 350, 150], [300, 0, 0, 0], [1000]*4, [True, False, False, False]),  # 锁定值越界
        SolveInput([250, 250, 350, 150], [600, 600, 0, 0], [1000]*4, [False]*4),  # 下限和超 1000
        SolveInput([250, 250, 350, 150], [0]*4, [200, 200, 200, 200], [False]*4),  # 上限和不足 1000
    ]
    check("非法输入均判 invalid", all(solve(forest, clf, c).status == INVALID for c in bad_cases))

    print("== 7. 求解失败路径 ==")
    import app.solver as sv
    orig_build = sv._build_optimize
    def boom(forest_, inp_):
        raise z3.Z3Exception("injected")
    sv._build_optimize = boom
    try:
        r_fail = solve(forest, clf, SolveInput([250, 250, 350, 150], [0]*4, [1000]*4, [False]*4))
    finally:
        sv._build_optimize = orig_build
    check("求解器异常 → solver_failure（非无解）", r_fail.status == SOLVER_FAILURE)

    print("== 8. API ==")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.get("/")
    check("GET / 返回网页", r.status_code == 200 and "开始求解" in r.text)
    r = client.get("/api/meta")
    mj = r.json()
    check("GET /api/meta", r.status_code == 200 and mj["n_trees"] == 5 and mj["max_depth"] == 3
          and abs(mj["proba_threshold"] - 0.7) < 1e-12)
    r = client.get("/api/samples")
    sj = r.json()
    check("GET /api/samples 两类样例齐全", r.status_code == 200
          and sj["feasible"]["check"]["status"] == "ok"
          and sj["infeasible"]["check"]["status"] == "infeasible")
    r = client.post("/api/solve", json={"original": fs["original"], "lower": fs["lower"],
                                        "upper": fs["upper"], "locked": fs["locked"]})
    rj = r.json()
    check("POST /api/solve 可行分支", r.status_code == 200 and rj["status"] == "ok"
          and rj["probability"] >= 0.7 and rj["suggestion"] == sug)
    r = client.post("/api/solve", json={"original": isf["original"], "lower": isf["lower"],
                                        "upper": isf["upper"], "locked": isf["locked"]})
    check("POST /api/solve 无解分支", r.json()["status"] == "infeasible")
    r = client.post("/api/solve", json={"original": [1, 2, 3, 4], "lower": [0]*4,
                                        "upper": [1000]*4, "locked": [False]*4})
    check("POST /api/solve 非法输入分支", r.json()["status"] == "invalid")
    html = client.get("/").text
    check("页面含迟到结果丢弃逻辑", "my !== epoch" in html and "abort" in html)

    print("== 9. 送检选样 ==")
    from fractions import Fraction
    import app.selection as selmod
    from app.selection import (
        LeafRegistry,
        SelectionInput,
        get_registry,
        reset_leaf_cache,
        route_formula,
        select,
    )
    reset_leaf_cache()
    reg = get_registry(forest)

    # 9.1 叶目录与 sklearn 训练样本数一致，收益精确等于 1/n_train
    sk_counts = []
    for est in clf.estimators_:
        t = est.tree_
        for nid in range(t.node_count):
            if t.children_left[nid] == t.children_right[nid]:
                sk_counts.append(int(t.n_node_samples[nid]))
    check("叶数与 sklearn 叶数一致", reg.n_leaves == len(sk_counts))
    check("n_train 与 sklearn 一致", sorted(x.n_train for x in reg.leaves) == sorted(sk_counts))
    check("叶收益精确为 1/n_train",
          all(x.weight == Fraction(1, x.n_train) for x in reg.leaves))

    # 9.2 路由掩码与 sklearn clf.apply（float32 输入）逐配方一致
    rng2 = np.random.default_rng(321)
    route_ok = True
    for _ in range(200):
        g = rng2.multinomial(TOTAL_GRAMS, [0.3, 0.3, 0.2, 0.2]).tolist()
        sk = clf.apply(np.array([[x / TOTAL_GRAMS for x in g]], dtype=np.float32))[0]
        expect = 0
        for ti, nid in enumerate(sk):
            expect |= 1 << reg.index[(ti, int(nid))]
        if route_formula(forest, g, reg) != expect:
            route_ok = False
            break
    check("叶路由与 sklearn apply 完全一致（float32）", route_ok)

    def rnd_formulas(rng, n):
        out, seen = [], set()
        while len(out) < n:
            g = rng.multinomial(TOTAL_GRAMS, [0.3, 0.3, 0.2, 0.2]).tolist()
            key = tuple(g)
            if key not in seen:
                seen.add(key)
                out.append(g)
        return out

    # 9.3 useful 样例：ok 且选中非空；收益精确核对；已检叶不出现；叶不重复
    us = samples["selection"]["useful"]
    ru = select(forest, SelectionInput(us["tested"], us["candidates"], us["slots"]))
    check("useful 样例返回 ok 且有推荐", ru.status == "ok" and len(ru.selected) >= 1)
    check("选中条数不超过名额", len(ru.selected) <= us["slots"])
    tested_mask = 0
    for row in us["tested"]:
        tested_mask |= route_formula(forest, row, reg)
    check("增量叶不含任何已检叶", (ru.gain_mask & tested_mask) == 0)
    check("增量叶无重复（位掩码天然去重）", len(ru.gain_leaves) == ru.gain_mask.bit_count())
    exact = sum((Fraction(1, l["n_train"]) for l in ru.gain_leaves), Fraction(0))
    check("收益 = 增量叶 1/n_train 之和（精确分数）",
          Fraction(ru.benefit_exact) == exact and abs(ru.benefit - float(exact)) < 1e-15)
    # 每条选中配方确实命中至少一个增量叶
    sel_masks = [route_formula(forest, row, reg) & ~tested_mask for row in ru.selected]
    check("每条推荐配方都贡献增量叶", all(m != 0 for m in sel_masks))
    check("增量并集与报告掩码一致",
          functools_reduce_or(sel_masks) == ru.gain_mask)

    # 9.4 同叶多条命中只计一次：10 条不同配方全部路由到同一掩码，名额 3
    # 任意非空组合收益完全相同 → 按“同分送检数最少”必须只推荐 1 条
    rng_dup = np.random.default_rng(77)
    dup_groups: dict[int, list] = {}
    target_n = 10
    shared_mask = None
    shared_rows: list = []
    while True:
        g = rng_dup.multinomial(TOTAL_GRAMS, [0.3, 0.3, 0.2, 0.2]).tolist()
        m = route_formula(forest, g, reg)
        bucket = dup_groups.setdefault(m, [])
        if all(g != x for x in bucket):
            bucket.append(g)
            if len(bucket) >= target_n:
                shared_mask, shared_rows = m, [list(x) for x in bucket]
                break
    r_dup = select(forest, SelectionInput([], shared_rows, 3))
    check("同掩码配方彼此不同", all(shared_rows[i] != shared_rows[j]
                                    for i in range(target_n) for j in range(i)))
    check("同叶去重：同掩码多条只计一次且同分取最少条数",
          r_dup.status == "ok" and len(r_dup.selected_indices) == 1
          and reg.mask_benefit(r_dup.gain_mask) == reg.mask_benefit(shared_mask))

    # 9.5 无增量 → 不推荐送检
    ng = samples["selection"]["no_gain"]
    rng3_res = select(forest, SelectionInput(ng["tested"], ng["candidates"], ng["slots"]))
    check("no_gain 样例 ok 且不推荐", rng3_res.status == "ok" and rng3_res.selected == []
          and rng3_res.benefit == 0.0)

    # 9.6 暴力枚举对照：最优收益 + 同分最少条数（多组随机数据、名额 1~5）
    brute_ok = True
    rng3 = np.random.default_rng(44)
    for trial in range(12):
        n_c = 10 + trial  # 10..21
        pool = rnd_formulas(rng3, n_c + 4)
        tested = [list(x) for x in pool[:4]]
        cands = [list(x) for x in pool[4:]]
        slots = 1 + trial % 5
        res = select(forest, SelectionInput(tested, cands, slots))
        tm = 0
        for row in tested:
            tm |= route_formula(forest, row, reg)
        cm = [route_formula(forest, row, reg) & ~tm for row in cands]
        best = None
        for k in range(1, min(slots, len(cm)) + 1):
            for combo in itertools.combinations(range(len(cm)), k):
                m = 0
                for i in combo:
                    m |= cm[i]
                b = reg.int_benefit(m)
                if best is None or b > best[0] or (b == best[0] and k < best[1]):
                    best = (b, k, combo)
        if (res.status != "ok" or reg.int_benefit(res.gain_mask) != best[0]
                or len(res.selected_indices) != best[1]):
            brute_ok = False
            break
    check("12 组随机数据暴力枚举对照最优性与同分规则", brute_ok)

    # 9.7 非法输入
    valid_c = rnd_formulas(np.random.default_rng(5), 10)
    bad_inputs = [
        SelectionInput([], valid_c[:9], 3),                 # 待检仅 9 条
        SelectionInput(rnd_formulas(np.random.default_rng(6), 21), valid_c, 3),  # 已检 21 条
        SelectionInput([], valid_c, 0),                     # 名额 0
        SelectionInput([], valid_c, 6),                     # 名额 6
        SelectionInput([], valid_c, True),                  # bool 名额
        SelectionInput([], [[1, 1, 1, 996]] * 10, 3),       # 总重 999
        SelectionInput([], [["1", 2, 3, 994]] + valid_c[1:], 3),  # 非整数
        SelectionInput([], valid_c + [valid_c[0]], 3),      # 同组重复
    ]
    check("非法选样输入均判 invalid",
          all(select(forest, x).status == "invalid" for x in bad_inputs))
    # 跨组重复允许（已检中出现过的配方仍可放在待检，只是增量为 0）
    r_cross = select(forest, SelectionInput([list(valid_c[0])], [list(x) for x in valid_c], 3))
    check("已检/待检跨组重复允许且正常求解", r_cross.status == "ok")

    # 9.8 预算耗尽 → pending（注入极小节点预算），不给出建议
    # 先收集掩码尽量分散的 60 条配方，保证穷举节点数必然超过预算检查点
    rng_big = np.random.default_rng(9)
    big, big_masks = [], set()
    while len(big) < 60:
        g = rng_big.multinomial(TOTAL_GRAMS, [0.3, 0.3, 0.2, 0.2]).tolist()
        m = route_formula(forest, g, reg)
        key = tuple(g)
        if key not in {tuple(x) for x in big}:
            big.append(g)
            big_masks.add(m)
    assert len(big_masks) >= 2
    orig_budget = selmod.NODE_BUDGET
    selmod.NODE_BUDGET = 50
    try:
        r_pend = select(forest, SelectionInput([], [list(x) for x in big], 5))
    finally:
        selmod.NODE_BUDGET = orig_budget
    check("预算耗尽返回 pending 且无推荐",
          r_pend.status == "pending" and r_pend.selected is None
          and not r_pend.exhaustive and r_pend.nodes_visited > 50)

    # 9.9 API：/select 页面、/api/leaves、/api/select 各分支
    r = client.get("/select")
    check("GET /select 返回选样网页", r.status_code == 200 and "送检" in r.text and "abort" in r.text)
    r = client.get("/api/leaves")
    lj = r.json()
    check("GET /api/leaves", r.status_code == 200 and lj["n_leaves"] == reg.n_leaves
          and all("weight" in x and "n_train" in x for x in lj["leaves"]))
    r = client.post("/api/select", json={"tested": us["tested"],
                                         "candidates": us["candidates"], "slots": us["slots"]})
    rj = r.json()
    check("POST /api/select ok 分支", r.status_code == 200 and rj["status"] == "ok"
          and len(rj["selected"]) >= 1 and rj["exhaustive"] is True)
    r = client.post("/api/select", json={"tested": ng["tested"],
                                         "candidates": ng["candidates"], "slots": ng["slots"]})
    check("POST /api/select 不推荐分支", r.json()["status"] == "ok"
          and r.json()["selected"] == [])
    r = client.post("/api/select", json={"tested": [], "candidates": valid_c[:9], "slots": 3})
    check("POST /api/select invalid 分支", r.json()["status"] == "invalid")
    check("选样样例由启动实测生成",
          sj["selection"]["useful"]["check"]["status"] == "ok"
          and sj["selection"]["no_gain"]["check"]["selected"] == [])

    print()
    if failures:
        print(f"共 {len(failures)} 项失败：{failures}")
        return 1
    print("全部自测通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
