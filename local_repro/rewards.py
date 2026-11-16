"""可验证奖励（RLVR）实现：与仓库 experiments/dapo/reward.py 逐函数对齐。

两套口径
--------
- **GRPO 口径**（experiments/grpo）：0/1 reward，只要最后一个 ``\\boxed{}`` 与标准答案
  字符串归一化后相同就给 1。
- **DAPO 口径**（experiments/dapo）：±1 reward（对 +1 / 错 -1），用 ``math_verify``
  做数学等价判定，并叠加 Soft Overlong Punishment。

本模块两种都给，默认走 DAPO 口径 —— 因为 ±1 的零中心 reward 更利于观察
group-relative advantage 的对称性，而等价判定能容纳 ``1/2`` 与 ``0.5`` 这类差异。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

# math_verify 的 verify() 内部用 multiprocessing + timeout 做安全校验，
# 在部分 Windows 环境下会抛 WinError 5/6（spawn 句柄无效）并返回 False。
# 因此默认关闭，改用下面的 sympy 判定；需要严格复刻仓库行为时设 ARL_USE_MATH_VERIFY=1。
USE_MATH_VERIFY = os.environ.get("ARL_USE_MATH_VERIFY", "0") == "1"

_HAS_MV = False
if USE_MATH_VERIFY:
    try:
        from math_verify import parse as _mv_parse
        from math_verify import verify as _mv_verify

        _HAS_MV = True
    except Exception:  # pragma: no cover - 环境相关
        _HAS_MV = False

HAS_MATH_VERIFY = _HAS_MV
ANSWER_WINDOW_CHARS = 300


# --------------------------------------------------------------------------- #
# 答案解析
# --------------------------------------------------------------------------- #
def extract_last_boxed(text: str) -> Optional[str]:
    """从末尾向前找到最后一个完整的 ``\\boxed{...}``，支持嵌套花括号。

    直接照搬仓库实现：从右往左找 ``\\boxed``，再做括号配对扫描。比正则
    ``\\\\boxed\\{([^}]+)\\}`` 更稳，因为答案里可能出现 ``\\frac{1}{2}`` 这种嵌套。
    """
    end = len(text)
    while True:
        marker = text.rfind("\\boxed", 0, end)
        if marker < 0:
            return None
        left = text.find("{", marker)
        if left < 0:
            end = marker
            continue
        depth = 0
        for position in range(left, len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
                if depth == 0:
                    return text[left + 1 : position].strip()
        end = marker


def normalize_answer(answer: str) -> str:
    """字符串级归一化：去空白、去千分位逗号、去尾随句号、统一小写。"""
    ans = answer.strip()
    ans = ans.replace(",", "").replace(" ", "").replace("$", "")
    ans = ans.rstrip(".。")
    return ans.lower()


_LATEX_RULES = [
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\!|\\,|\\;|\\ ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"\\cdot|\\times", "*"),
    (r"\\div", "/"),
    (r"\\pi", "pi"),
]


def _latex_to_py(expr: str) -> str:
    """把常见的 LaTeX 片段转成 sympy 可解析的 Python 表达式。

    只处理 GSM8K / MATH 里高频的那几种：``\\frac{a}{b}``、``\\sqrt{}``、
    ``^{}``、百分号、千分位逗号。目标是**确定性判定**，不追求覆盖全部 LaTeX。
    """
    s = expr.strip().replace("$", "").replace(",", "")
    for pat, rep in _LATEX_RULES:
        s = re.sub(pat, rep, s)
    # 反复替换，处理 \frac{\frac{1}{2}}{3} 这类嵌套
    for _ in range(5):
        new = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
        if new == s:
            break
        s = new
    for _ in range(3):
        new = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
        if new == s:
            break
        s = new
    s = s.replace("\\%", "/100").replace("%", "/100")
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\", "")
    return s.strip()


def _numeric_value(expr: str) -> Optional[float]:
    """尝试把表达式解析成浮点数：先直读，再 sympy 化简。"""
    py = _latex_to_py(expr)
    try:
        return float(py)
    except Exception:
        pass
    try:
        import sympy

        val = sympy.sympify(py, rational=True)
        if val.free_symbols:
            return None
        return float(val.evalf())
    except Exception:
        return None


def answers_equivalent(prediction: str, reference: str) -> bool:
    """数学等价判定。

    优先级：math_verify（可选）→ 字符串归一化 → sympy 数值等价。
    """
    if _HAS_MV:
        try:
            return bool(_mv_verify(_mv_parse(f"${reference}$"), _mv_parse(f"${prediction}$")))
        except Exception:
            pass

    if normalize_answer(prediction) == normalize_answer(reference):
        return True

    pred_val = _numeric_value(prediction)
    ref_val = _numeric_value(reference)
    if pred_val is not None and ref_val is not None:
        return abs(pred_val - ref_val) <= 1e-6 * max(1.0, abs(ref_val))

    # 符号等价：把差化简为 0
    try:
        import sympy

        diff = sympy.sympify(_latex_to_py(prediction), rational=True) - sympy.sympify(
            _latex_to_py(reference), rational=True
        )
        return bool(sympy.simplify(diff) == 0)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 奖励
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RewardResult:
    """一条 completion 的可解释奖励拆解。"""

    base_reward: float
    length_penalty: float
    shaped_reward: float
    correct: bool
    valid_format: bool
    answer: Optional[str]
    completion_tokens: int


def soft_overlong_penalty_value(
    completion_tokens: int, *, max_tokens: int, overlong_cache: int
) -> float:
    """DAPO 分段线性长度惩罚，返回 ``[-1, 0]``。"""
    if completion_tokens < 0:
        raise ValueError("completion_tokens must be >= 0")
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if not 1 <= overlong_cache <= max_tokens:
        raise ValueError("overlong_cache must be in [1, max_tokens]")

    penalty_start = max_tokens - overlong_cache
    if completion_tokens <= penalty_start:
        return 0.0
    if completion_tokens <= max_tokens:
        return -(completion_tokens - penalty_start) / overlong_cache
    return -1.0


def score_answer(
    text: str,
    reference: str,
    *,
    completion_tokens: int,
    max_tokens: int = 4096,
    overlong_cache: int = 1024,
    enable_overlong: bool = True,
    mode: str = "pm1",
) -> RewardResult:
    """给一条 completion 打分。

    Parameters
    ----------
    mode
        - ``"pm1"``：DAPO 口径，对 +1 / 错 -1；
        - ``"01"`` ：GRPO 口径，对 1 / 错 0。
    enable_overlong
        是否叠加 DAPO 的软超长惩罚（GRPO 章节里为 False）。
    """
    answer = extract_last_boxed(text[-ANSWER_WINDOW_CHARS:])
    correct = answer is not None and answers_equivalent(answer, reference)

    if mode == "01":
        base = 1.0 if correct else 0.0
    elif mode == "pm1":
        base = 1.0 if correct else -1.0
    else:
        raise ValueError(f"unknown reward mode: {mode}")

    penalty = (
        soft_overlong_penalty_value(
            completion_tokens, max_tokens=max_tokens, overlong_cache=overlong_cache
        )
        if enable_overlong
        else 0.0
    )
    return RewardResult(
        base_reward=base,
        length_penalty=penalty,
        shaped_reward=base + penalty,
        correct=correct,
        valid_format=answer is not None,
        answer=answer,
        completion_tokens=completion_tokens,
    )


def format_reward(text: str) -> float:
    """格式奖励（Search-R1 / Spec-o3 章节用到的最小版本）。

    只检查是否同时出现思考块与答案块，用于在答案正确性之外额外塑造格式。
    """
    has_think = "<think>" in text and "</think>" in text
    has_answer = extract_last_boxed(text) is not None
    return float(has_think) * 0.5 + float(has_answer) * 0.5


__all__ = [
    "ANSWER_WINDOW_CHARS",
    "HAS_MATH_VERIFY",
    "RewardResult",
    "answers_equivalent",
    "extract_last_boxed",
    "format_reward",
    "normalize_answer",
    "score_answer",
    "soft_overlong_penalty_value",
]
