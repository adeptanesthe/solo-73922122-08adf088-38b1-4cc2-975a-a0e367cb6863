"""可行 / 无解样例：启动时用真实模型与求解器验证后生成，保证与当前模型一致。"""
from __future__ import annotations

import numpy as np

from .model import FEATURE_NAMES, predict_proba_qualified
from .selection import SelectionInput, get_registry, route_formula, select
from .solver import INFEASIBLE, OK, SolveInput, solve

# 可行样例候选：原配方不合格，但不难通过少量改动达标
_FEASIBLE_CANDIDATES = [
    # (原配方, 下限, 上限, 锁定)
    ([250, 250, 350, 150], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [False, True, False, False]),
    ([300, 200, 350, 150], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [False, False, False, False]),
    ([200, 300, 300, 200], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [True, False, False, False]),
    ([350, 150, 350, 150], [100, 0, 0, 0], [1000, 1000, 600, 1000], [False, False, False, False]),
]

# 无解样例候选：限制条件把配方逼进不可能合格的区域（非简单全锁）
_INFEASIBLE_CANDIDATES = [
    # 填料(原料3)被迫 >= 620g，且再生PP(原料1)被压到 <= 150g
    ([150, 130, 620, 100], [0, 0, 620, 0], [150, 1000, 1000, 1000], [False, False, False, False]),
    # 全部锁定在一个不合格配方上
    ([150, 130, 620, 100], [0, 0, 0, 0], [1000, 1000, 1000, 1000], [True, True, True, True]),
    ([200, 150, 550, 100], [0, 0, 550, 0], [200, 1000, 1000, 1000], [False, False, False, False]),
]


def _try_candidates(forest, clf, candidates, want_status):
    for original, lower, upper, locked in candidates:
        inp = SolveInput(original=list(original), lower=list(lower), upper=list(upper), locked=list(locked))
        res = solve(forest, clf, inp)
        if res.status == want_status:
            return {
                "original": list(original),
                "lower": list(lower),
                "upper": list(upper),
                "locked": list(locked),
                "original_probability": predict_proba_qualified(clf, list(original)),
                "check": res.to_dict(),
            }
    return None


def build_samples(forest, clf) -> dict:
    """返回 {"feasible": ..., "infeasible": ...}，均经求解器实测。"""
    feasible = _try_candidates(forest, clf, _FEASIBLE_CANDIDATES, OK)
    infeasible = _try_candidates(forest, clf, _INFEASIBLE_CANDIDATES, INFEASIBLE)
    if feasible is None or infeasible is None:
        raise RuntimeError("样例生成失败：候选样例未通过求解器验证")
    selection = build_selection_sample(forest)
    return {
        "feasible": feasible,
        "infeasible": infeasible,
        "selection": selection,
        "feature_names": FEATURE_NAMES,
    }


def _random_formula(rng) -> list[int]:
    """总重 1000 的随机整数四料配方（Dirichlet 取整后配平）。"""
    p = rng.dirichlet([2.5, 2.0, 1.6, 1.2])
    g = [int(round(x * 1000)) for x in p]
    g[0] += 1000 - sum(g)
    return g


def build_selection_sample(forest) -> dict:
    """构造送检选样的演示数据，并经真实选样器实测。

    useful：已检配方只覆盖部分叶，待检配方存在增量收益，返回含推荐的 ok 结果；
    no_gain：已检配方覆盖了待检池的全部可达叶，返回“不推荐送检”的 ok 结果。
    """
    reg = get_registry(forest)
    rng = np.random.default_rng(7701)

    # ---- useful 样例 ----
    useful = None
    for _ in range(200):
        pool, seen = [], set()
        while len(pool) < 30:
            f = _random_formula(rng)
            key = tuple(f)
            if key not in seen:
                seen.add(key)
                pool.append(f)
        tested, candidates = pool[:5], pool[5:]
        inp = SelectionInput(tested=[list(x) for x in tested],
                             candidates=[list(x) for x in candidates], slots=3)
        res = select(forest, inp)
        if res.status == "ok" and res.selected:
            useful = {
                "tested": [list(x) for x in tested],
                "candidates": [list(x) for x in candidates],
                "slots": 3,
                "check": res.to_dict(),
            }
            break
    if useful is None:
        raise RuntimeError("送检选样 useful 样例生成失败")

    # ---- no_gain 样例：已检集合覆盖待检池全部可达叶 ----
    no_gain = None
    rng2 = np.random.default_rng(8802)
    for _ in range(500):
        pool, seen = [], set()
        while len(pool) < 24:
            f = _random_formula(rng2)
            key = tuple(f)
            if key not in seen:
                seen.add(key)
                pool.append(f)
        tested, candidates = [], pool[12:]
        tested_mask = 0
        for f in pool[:12]:
            m = route_formula(forest, f, reg)
            if m & ~tested_mask:
                tested.append(list(f))
                tested_mask |= m
        cand_union = 0
        for f in candidates:
            cand_union |= route_formula(forest, f, reg)
        if cand_union & ~tested_mask == 0 and len(tested) <= 20 and len(candidates) >= 10:
            inp2 = SelectionInput(tested=[list(x) for x in tested],
                                  candidates=[list(x) for x in candidates], slots=3)
            res2 = select(forest, inp2)
            if res2.status == "ok" and not res2.selected:
                no_gain = {
                    "tested": [list(x) for x in tested],
                    "candidates": [list(x) for x in candidates],
                    "slots": 3,
                    "check": res2.to_dict(),
                }
                break
    if no_gain is None:
        raise RuntimeError("送检选样 no_gain 样例生成失败")
    return {"useful": useful, "no_gain": no_gain}
