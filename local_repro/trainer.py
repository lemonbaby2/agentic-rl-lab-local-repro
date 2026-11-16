"""本地 GRPO / DAPO / GSPO 训练器 —— 用 HF + peft 替代 PyTRIO 云端 Train API。

为什么需要这个文件
------------------
原仓库把「前向、反向、优化器更新、LoRA 权重保存、采样服务」全部交给 PyTRIO 云端，
本地只跑数据、reward 和 loss。要在**本机 GPU** 上复现，就必须补上被交出去的那一半：

    PyTRIO 云端              本文件对应的本地实现
    ----------------------   ------------------------------------------
    base 模型 + LoRA         transformers AutoModelForCausalLM + peft
    sampling client          model.generate(num_return_sequences=G)
    逐 token logprob         一次 teacher-forcing 前向 + cross_entropy
    forward_backward         loss.backward() + optimizer.step()
    save_weights_for_sampler 直接复用同一份 LoRA 权重（无需跨机同步）

算法层面保持与仓库一致：group rollout → 组内相对 advantage → token/completion 掩码
→ PPO/CISPO/GSPO/DAPO 更新。差别只在「算力在哪」。
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import losses as L
from .data import MathProblem, build_prompt, build_prompt_text
from .rewards import extract_last_boxed, score_answer


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """一次训练实验的全部开关，与仓库 CLI 参数一一对应。"""

    # 算法
    algorithm: str = "grpo_ppo"  # 见 losses.PRESETS
    reward_mode: str = "pm1"  # pm1: DAPO 口径(±1)，01: GRPO 口径(0/1)
    normalize_std: bool = False  # 论文版 GRPO 会除以组内 std
    ppo_epochs: int = 1  # 一轮 rollout 内做几次梯度更新（>1 时 ratio 才会偏离 1）
    # 动态采样（DAPO）
    dynamic_sampling: bool = False
    max_candidate_multiplier: int = 2
    # 软超长惩罚（DAPO）
    enable_overlong: bool = False
    max_tokens: int = 512
    overlong_cache: int = 128
    # 模型
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # 采样
    steps: int = 15
    groups_per_step: int = 4
    group_size: int = 4
    max_new_tokens: int = 256
    max_prompt_tokens: int = 512
    prompt_style: str = "plain"  # plain: Qwen2.5 适配版；repo: 仓库原版(<think> 标签)
    temperature: float = 1.0
    top_p: float = 0.95
    # 优化
    learning_rate: float = 1e-5
    micro_batch_size: int = 2
    grad_clip: float = 1.0
    seed: int = 42
    # 数据集规模（写进日志，方便判断两次运行是否可比）
    train_limit: int = 0  # 0 表示用全部
    dev_limit: int = 0
    train_size: int = 0  # 实际过滤后可用条数，运行时回填
    dev_size: int = 0
    # 记录
    tag: str = "run"
    out_dir: str = "logs"


@dataclass
class Rollout:
    """一条 completion 及其训练所需全部张量。"""

    prompt_ids: List[int]
    completion_ids: List[int]
    text: str
    reward: float
    base_reward: float
    correct: bool
    advantage: float = 0.0


@dataclass
class RolloutGroup:
    """同一道题的一组采样，advantage 在组内计算。"""

    problem: MathProblem
    rollouts: List[Rollout] = field(default_factory=list)

    @property
    def rewards(self) -> List[float]:
        return [r.reward for r in self.rollouts]

    @property
    def degenerate(self) -> bool:
        """原始正确性信号是否退化（全对或全错）—— DAPO 判断口径。"""
        flags = {r.correct for r in self.rollouts}
        return len(flags) <= 1


# --------------------------------------------------------------------------- #
# 训练器
# --------------------------------------------------------------------------- #
class LocalGRPOTrainer:
    """单卡可跑的 GRPO 家族训练器。"""

    def __init__(self, cfg: TrainConfig, problems: List[MathProblem], dev: Optional[List[MathProblem]] = None):
        self.cfg = cfg
        self.problems = problems
        self.dev = dev or []
        self.rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.history: List[Dict[str, Any]] = []
        self._cursor = 0  # 按顺序取题，避免同一步重复出题

        self._load_model()
        self.loss_cfg = L.PRESETS[cfg.algorithm]
        self._prepare_dirs()

    # ---------------- 模型 ----------------
    def _load_model(self) -> None:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.cfg
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            cfg.dtype
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, dtype=dtype, trust_remote_code=True
        )
        lora_cfg = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.to(self.device)
        self.model = model
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"[model] {cfg.model_name} | trainable {trainable/1e6:.2f}M / total {total/1e6:.1f}M "
            f"({100*trainable/total:.2f}%) | device {cfg.device} | dtype {cfg.dtype}"
        )

    def _prepare_dirs(self) -> None:
        Path(self.cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # ---------------- 采样 ----------------
    def _build_prompt(self, problem: MathProblem) -> List[int]:
        ids = build_prompt(self.tokenizer, problem.question, style=self.cfg.prompt_style)
        return ids[-self.cfg.max_prompt_tokens :]

    @torch.no_grad()
    def _sample_group(self, problem: MathProblem) -> RolloutGroup:
        """对一道题采样 G 条 completion（等价于仓库的 sampling client 调用）。"""
        cfg = self.cfg
        prompt_ids = self._build_prompt(problem)
        input_ids = torch.tensor([prompt_ids], device=self.device)
        attn = torch.ones_like(input_ids)
        self.model.eval()
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            num_return_sequences=cfg.group_size,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        group = RolloutGroup(problem=problem)
        prompt_len = len(prompt_ids)
        for seq in out:
            completion_ids = seq[prompt_len:].tolist()
            # 去掉 eos 之后的填充
            if self.tokenizer.eos_token_id in completion_ids:
                completion_ids = completion_ids[: completion_ids.index(self.tokenizer.eos_token_id) + 1]
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            res = score_answer(
                text,
                problem.answer,
                completion_tokens=len(completion_ids),
                max_tokens=cfg.max_tokens,
                overlong_cache=cfg.overlong_cache,
                enable_overlong=cfg.enable_overlong,
                mode=cfg.reward_mode,
            )
            group.rollouts.append(
                Rollout(
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    text=text,
                    reward=res.shaped_reward,
                    base_reward=res.base_reward,
                    correct=res.correct,
                )
            )
        return group

    # ---------------- logprob ----------------
    def _batch_tensors(self, rollouts: List[Rollout]) -> Dict[str, torch.Tensor]:
        """右填充拼 batch，并算出 completion 掩码。"""
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(r.prompt_ids) + len(r.completion_ids) for r in rollouts)
        input_ids, attn, prompt_lens = [], [], []
        for r in rollouts:
            ids = r.prompt_ids + r.completion_ids
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            prompt_lens.append(len(r.prompt_ids))
        return {
            "input_ids": torch.tensor(input_ids, device=self.device),
            "attention_mask": torch.tensor(attn, device=self.device),
            "prompt_lens": torch.tensor(prompt_lens, device=self.device),
        }

    def _token_logprobs(
        self, batch: Dict[str, torch.Tensor], grad: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 completion 区间的逐 token logprob 与掩码。

        用 ``cross_entropy(reduction="none")`` 拿 ``-logp``，比手工
        ``log_softmax + gather`` 省显存（V=151936 时差别很大）。
        """
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = batch["input_ids"][:, 1:].clone()

            # 只保留 completion 区间：第 j 个 logits 预测第 j+1 个 token
            positions = torch.arange(shift_labels.size(1), device=self.device).unsqueeze(0)
            mask = (positions >= (batch["prompt_lens"].unsqueeze(1) - 1)) & (
                batch["attention_mask"][:, 1:].bool()
            )
            shift_labels[~mask] = -100

            ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                shift_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).reshape(shift_labels.shape)
            logp = -ce
        return logp, mask.float()

    # ---------------- 训练 ----------------
    def _next_problems(self, n: int) -> List[MathProblem]:
        picked = []
        for _ in range(n):
            if self._cursor >= len(self.problems):
                self._cursor = 0
                self.rng.shuffle(self.problems)
            picked.append(self.problems[self._cursor])
            self._cursor += 1
        return picked

    def _collect_groups(self) -> Tuple[List[RolloutGroup], int]:
        """采一批 group；DAPO 下做动态采样过滤。"""
        cfg = self.cfg
        groups: List[RolloutGroup] = []
        candidates = 0
        budget = cfg.groups_per_step * (cfg.max_candidate_multiplier if cfg.dynamic_sampling else 1)
        pool = self._next_problems(budget)
        for problem in pool:
            if len(groups) >= cfg.groups_per_step:
                break
            candidates += 1
            g = self._sample_group(problem)
            if cfg.dynamic_sampling and g.degenerate:
                continue  # 丢掉全对/全错的 group，继续补采
            groups.append(g)
        return groups, candidates - len(groups)

    def _assign_advantages(self, groups: List[RolloutGroup]) -> None:
        """组内相对 advantage，写回每条 rollout。"""
        cfg = self.cfg
        rewards = torch.tensor(
            [r.reward for g in groups for r in g.rollouts], dtype=torch.float32, device=self.device
        )
        adv, _ = L.group_relative_advantage(
            rewards, cfg.group_size, normalize_std=cfg.normalize_std
        )
        idx = 0
        for g in groups:
            for r in g.rollouts:
                r.advantage = float(adv[idx].item())
                idx += 1

    def train(self) -> List[Dict[str, Any]]:
        cfg = self.cfg
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, betas=(0.9, 0.999))
        loss_kwargs = dict(
            clip_low=self.loss_cfg.clip_low,
            clip_high=self.loss_cfg.clip_high,
            reduction=self.loss_cfg.reduction,
        )
        if self.loss_cfg.name == "gspo":
            loss_kwargs = {"clip_eps": self.loss_cfg.clip_eps}
        loss_fn = L.build_loss(self.loss_cfg.name, **loss_kwargs)

        base_acc = self.evaluate(self.dev) if self.dev else {}
        print(f"[eval] base {base_acc}")

        for step in range(1, cfg.steps + 1):
            t0 = time.time()
            groups, filtered = self._collect_groups()
            if not groups:
                print(f"[step {step}] 全部 group 退化，跳过")
                continue
            self._assign_advantages(groups)
            torch.cuda.empty_cache()

            flat: List[Rollout] = [r for g in groups for r in g.rollouts]
            adv_vec = torch.tensor([r.advantage for r in flat], device=self.device, dtype=torch.float32)

            # 分块（micro-batch）处理：152k 词表的 cross_entropy 张量很大，
            # 一次前向放不下 12 条 400+ token 的序列，所以按 micro_batch_size 切块，
            # 每块单独前向 + 反向，梯度累积后统一步一次。
            chunks: List[List[int]] = [
                list(range(i, min(i + cfg.micro_batch_size, len(flat))))
                for i in range(0, len(flat), cfg.micro_batch_size)
            ]

            # 旧策略 logprob 在整个 step 内固定，逐块预先算好（no_grad）
            chunk_cache = []
            for idxs in chunks:
                sub_batch = self._batch_tensors([flat[i] for i in idxs])
                lp, mk = self._token_logprobs(sub_batch, grad=False)
                chunk_cache.append((sub_batch, lp.detach(), mk, idxs))

            total_tokens = sum(c[2].sum() for c in chunk_cache)
            total_seqs = len(flat)
            len_tok = torch.cat([c[2].sum(dim=1).cpu() for c in chunk_cache])
            entropy_est = float(
                sum((-(c[1]) * c[2]).sum().item() for c in chunk_cache) / max(total_tokens.item(), 1)
            )

            step_metrics: Dict[str, float] = {}
            # ppo_epochs > 1 时，从第二轮开始 ratio 才会明显偏离 1，
            # 这正是 PPO/CISPO 裁剪开始起作用、也是观察 clip_frac 的时机。
            for epoch in range(cfg.ppo_epochs):
                self.model.train()
                optimizer.zero_grad(set_to_none=True)
                epoch_metrics: Dict[str, float] = {}
                for sub_batch, old_lp, mk, idxs in chunk_cache:
                    adv_chunk = adv_vec[idxs].unsqueeze(1).expand_as(old_lp) * mk
                    new_lp, new_mk = self._token_logprobs(sub_batch, grad=True)
                    loss, lm = loss_fn(new_lp, old_lp, adv_chunk, new_mk)
                    # 还原成全局归一化：token mean 按 token 数加权，sequence mean 按序列数加权
                    if self.loss_cfg.name == "gspo" or self.loss_cfg.reduction == "sequence":
                        weight = len(idxs) / total_seqs
                    else:
                        weight = mk.sum() / total_tokens
                    (loss * weight).backward()
                    for k, v in lm.items():
                        epoch_metrics.setdefault(k, 0.0)
                        epoch_metrics[k] += v * (
                            (mk.sum().item() / total_tokens.item())
                            if self.loss_cfg.reduction != "sequence"
                            else len(idxs) / total_seqs
                        )
                    epoch_metrics["loss"] = epoch_metrics.get("loss", 0.0) + float(loss.item()) * (
                        (mk.sum().item() / total_tokens.item())
                        if self.loss_cfg.reduction != "sequence"
                        else len(idxs) / total_seqs
                    )
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                optimizer.step()
                for k, v in epoch_metrics.items():
                    step_metrics[f"{k}_e{epoch}"] = v
                step_metrics["loss"] = epoch_metrics.get("loss", 0.0)

            rewards = [r.reward for r in flat]
            corr = [float(r.correct) for r in flat]
            fmt = [float(extract_last_boxed(r.text) is not None) for r in flat]
            rec = {
                "step": step,
                "reward_mean": sum(rewards) / len(rewards),
                "accuracy": sum(corr) / len(corr),
                "format_rate": sum(fmt) / len(fmt),
                "length_mean": float(len_tok.float().mean().item()),
                "advantage_std": float(adv_vec.std().item()) if adv_vec.numel() > 1 else 0.0,
                "degenerate_frac": sum(float(g.degenerate) for g in groups) / len(groups),
                "filtered_groups": filtered,
                "policy_entropy": entropy_est,
                "sec": round(time.time() - t0, 2),
                **step_metrics,
            }
            self.history.append(rec)
            print(
                f"[{cfg.tag}] step {step:02d} | reward {rec['reward_mean']:+.3f} | acc {rec['accuracy']:.3f} "
                f"| len {rec['length_mean']:.0f} | H {rec['policy_entropy']:.2f} "
                f"| loss {rec.get('loss', 0):+.4f} | {rec['sec']}s"
            )

        final = self.evaluate(self.dev) if self.dev else {}
        out = {
            "config": asdict(cfg),
            "base_eval": base_acc,
            "final_eval": final,
            "history": self.history,
        }
        path = Path(cfg.out_dir) / f"{cfg.tag}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[save] {path}")
        return self.history

    # ---------------- 评测 ----------------
    @torch.no_grad()
    def evaluate(self, problems: List[MathProblem], max_items: int = 64) -> Dict[str, float]:
        """贪心解码评测准确率（对应仓库的 eval.py / 评测 sampler weights）。"""
        if not problems:
            return {}
        self.model.eval()
        subset = problems[:max_items]
        correct = 0
        for p in subset:
            prompt_ids = self._build_prompt(p)
            ids = torch.tensor([prompt_ids], device=self.device)
            out = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            text = self.tokenizer.decode(out[0][len(prompt_ids) :], skip_special_tokens=True)
            res = score_answer(
                text, p.answer, completion_tokens=1, enable_overlong=False, mode=self.cfg.reward_mode
            )
            correct += int(res.correct)
        return {"accuracy": correct / len(subset), "n": len(subset)}

    def save_adapter(self, path: str) -> None:
        """只保存 LoRA 适配器（对应 save_weights_for_sampler）。"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[save] adapter -> {path}")


__all__ = ["LocalGRPOTrainer", "Rollout", "RolloutGroup", "TrainConfig"]
