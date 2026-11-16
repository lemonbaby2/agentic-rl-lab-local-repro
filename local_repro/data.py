"""GSM8K / DAPO-Math 数据准备：与仓库 experiments/grpo 的 prompt 构造完全一致。

prompt 结构 = few-shot 示例 + 用户问题 + ``\\boxed{}`` 格式要求，
chat template 用 Qwen 系列自带的 template，关闭 thinking 以缩短输出。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List

QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."

# 仓库原版 few-shot：一个"数 r 的个数"的例子，用来稳定 \boxed{} 输出格式。
# 注意它带 <think> 标签 —— 这是 Qwen3 系列的思考块约定，Qwen2.5 不认。
FEWSHOT_PREFIX: List[Dict[str, str]] = [
    {"role": "user", "content": "How many r's are in strawberry?" + QUESTION_SUFFIX},
    {
        "role": "assistant",
        "content": (
            "<think>\n\n</think>\n\n"
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. "
            "There are three r's. \\boxed{3}"
        ),
    },
]

# Qwen2.5 适配版：去掉 <think> 标签，换成一道典型的 GSM8K 两步题做示范。
# 本机实测（10 道 dev 题、贪心解码）：适配版 4/10 正确、7/10 格式合规；
# 原版 0/10 正确、0/10 格式合规 —— 原版 few-shot 与基座的模板约定不匹配。
FEWSHOT_PREFIX_PLAIN: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": "Natalia sold clips to 48 friends in April, and then she sold half as many "
        "clips in May. How many clips did she sell altogether?" + QUESTION_SUFFIX,
    },
    {
        "role": "assistant",
        "content": "In April she sold 48. In May she sold 48/2 = 24. "
        "Altogether 48 + 24 = 72. \\boxed{72}",
    },
]

PROMPT_STYLES = {"repo": FEWSHOT_PREFIX, "plain": FEWSHOT_PREFIX_PLAIN}


@dataclass
class MathProblem:
    """一道题：原始问题 + 参考答案。"""

    question: str
    answer: str
    source: str = "gsm8k"


def load_gsm8k(split: str = "train", limit: int | None = None, seed: int = 42) -> List[MathProblem]:
    """加载 GSM8K，抽固定子集。

    只保留 0.5B 模型**做得动**的题（答案 ≤ 3 位数、题干不超过 400 字符），
    这样本地小算力也能观察到 reward 上升，而不是全程 0 奖励。
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    problems: List[MathProblem] = []
    for row in ds:
        raw_answer = row["answer"].split("####")[-1].strip().replace(",", "")
        if not raw_answer.isdigit():
            continue
        if len(row["question"]) > 400 or len(raw_answer) > 3:
            continue
        problems.append(MathProblem(question=row["question"].strip(), answer=raw_answer))

    rng = random.Random(seed)
    rng.shuffle(problems)
    if limit is not None:
        problems = problems[:limit]
    return problems


def build_prompt(tokenizer: Any, question: str, fewshot: bool = True, style: str = "plain") -> List[int]:
    """把问题渲染成 chat 模板并 tokenize。

    ``style`` 决定 few-shot 用哪一套：``"repo"`` 复刻仓库原版（含 ``<think>`` 标签），
    ``"plain"`` 是 Qwen2.5 适配版。
    """
    prefix = PROMPT_STYLES.get(style, FEWSHOT_PREFIX_PLAIN)
    messages = list(prefix) if fewshot else []
    messages.append({"role": "user", "content": question + QUESTION_SUFFIX})
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(prompt_text, add_special_tokens=False)


def build_prompt_text(tokenizer: Any, question: str, fewshot: bool = True, style: str = "plain") -> str:
    """同上但返回字符串，便于调试与可视化。"""
    prefix = PROMPT_STYLES.get(style, FEWSHOT_PREFIX_PLAIN)
    messages = list(prefix) if fewshot else []
    messages.append({"role": "user", "content": question + QUESTION_SUFFIX})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


__all__ = [
    "FEWSHOT_PREFIX",
    "FEWSHOT_PREFIX_PLAIN",
    "MathProblem",
    "PROMPT_STYLES",
    "QUESTION_SUFFIX",
    "build_prompt",
    "build_prompt_text",
    "load_gsm8k",
]
