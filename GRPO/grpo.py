import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# 配置
# =========================

@dataclass
class GRPOConfig:
    model_name: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    max_prompt_length: int = 128
    max_new_tokens: int = 64

    batch_size: int = 2          # prompt 数
    group_size: int = 4          # 每个 prompt 采样几条回答

    lr: float = 1e-5
    clip_eps: float = 0.2
    kl_coef: float = 0.02
    temperature: float = 1.0

    pad_token_id: int = 50256    # gpt2 默认 eos 兼 pad
    eos_token_id: int = 50256

    train_steps: int = 50


# =========================
# 一个非常简单的 reward 函数示例
# 实际使用时你会替换成:
# - reward model
# - verifier
# - 规则判分
# - 程序测试器
# =========================

def simple_reward_fn(prompt: str, response: str) -> float:
    """
    这里只是示例：
    - 如果回答里包含 "4"，给高分
    - 太长略微惩罚
    只是为了演示 GRPO 流程，别把它当真实 reward 设计。
    """
    score = 0.0

    if "2+2" in prompt:
        if "4" in response:
            score += 1.0
        else:
            score -= 1.0

    # 很轻的长度惩罚，避免无脑变长
    score -= 0.002 * len(response)

    return score


# =========================
# 数据：你可以换成自己的 prompt 数据集
# =========================

PROMPTS = [
    "Question: 2+2等于几？\nAnswer:",
    "Question: 请简短回答：2+2是多少？\nAnswer:",
    "Question: 用一句话回答，2+2=?\nAnswer:",
    "Question: 直接告诉我 2+2 的结果。\nAnswer:",
]


# =========================
# 工具函数
# =========================

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    denom = mask.sum(dim=dim).clamp(min=1)
    return (x * mask).sum(dim=dim) / denom


def group_normalize(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    rewards: [batch_size, group_size]
    return : [batch_size, group_size]
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    adv = (rewards - mean) / (std + eps)
    return adv


def sample_prompts(all_prompts: List[str], batch_size: int) -> List[str]:
    idx = torch.randint(0, len(all_prompts), (batch_size,))
    return [all_prompts[i] for i in idx.tolist()]


# =========================
# 采样：同一个 prompt 生成 group 条回答
# =========================

@torch.no_grad()
def generate_group_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    cfg: GRPOConfig,
) -> Dict[str, List]:
    """
    返回结构：
    {
        "prompt_texts": [...],                       长度 B
        "response_texts": [[...]*G for _ in B],     B x G
        "prompt_input_ids": [tensor],               长度 B，每个形状 [prompt_len]
        "full_sequences": [[tensor]*G for _ in B],  B x G，每个形状 [seq_len]
        "response_masks": [[tensor]*G for _ in B],  B x G，每个形状 [seq_len-1]
    }
    response_mask 的含义：
      它对“需要训练的 token 位置”标 1，只覆盖 response 区域对应的 label 位置。
    """
    model.eval()

    prompt_input_ids = []
    response_texts = []
    full_sequences = []
    response_masks = []

    for prompt in prompts:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_prompt_length,
        )
        input_ids = enc["input_ids"][0].to(cfg.device)
        prompt_input_ids.append(input_ids)

        group_texts = []
        group_seqs = []
        group_masks = []

        for _ in range(cfg.group_size):
            out = model.generate(
                input_ids=input_ids.unsqueeze(0),
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                pad_token_id=cfg.pad_token_id,
                eos_token_id=cfg.eos_token_id,
            )[0]

            prompt_len = input_ids.size(0)
            seq_len = out.size(0)

            # decode response 文本
            response_ids = out[prompt_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            # response_mask 是对 labels 对齐后的 mask
            # logits[:, :-1] 预测 labels[:, 1:]
            # 若某 token 属于 response，那么它在 labels 中的下标是 >= prompt_len
            # 对应到 shift 后的位置是 [prompt_len-1, seq_len-2]
            mask = torch.zeros(seq_len - 1, dtype=torch.float32, device=cfg.device)
            if seq_len - 1 > 0 and prompt_len - 1 < seq_len - 1:
                mask[prompt_len - 1:] = 1.0

            group_texts.append(response_text)
            group_seqs.append(out.to(cfg.device))
            group_masks.append(mask)

        response_texts.append(group_texts)
        full_sequences.append(group_seqs)
        response_masks.append(group_masks)

    return {
        "prompt_texts": prompts,
        "response_texts": response_texts,
        "prompt_input_ids": prompt_input_ids,
        "full_sequences": full_sequences,
        "response_masks": response_masks,
    }


# =========================
# reward 计算
# =========================

def compute_group_rewards(
    prompt_texts: List[str],
    response_texts: List[List[str]],
    reward_fn: Callable[[str, str], float],
    device: str,
) -> torch.Tensor:
    """
    return: [B, G]
    """
    rewards = []
    for prompt, group in zip(prompt_texts, response_texts):
        rewards.append([reward_fn(prompt, resp) for resp in group])
    return torch.tensor(rewards, dtype=torch.float32, device=device)


# =========================
# 计算给定序列上 token 的 logprob
# =========================

def compute_sequence_token_logprobs(
    model: AutoModelForCausalLM,
    seq: torch.Tensor,
) -> torch.Tensor:
    """
    seq: [seq_len]
    return: [seq_len - 1]
      每个位置对应 labels[:,1:] 中那个 token 的 logprob
    """
    outputs = model(input_ids=seq.unsqueeze(0))
    logits = outputs.logits[:, :-1, :]               # [1, seq_len-1, vocab]
    labels = seq.unsqueeze(0)[:, 1:]                 # [1, seq_len-1]

    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1).squeeze(0)                         # [seq_len-1]

    return token_logprobs


def kl_from_logprobs(
    current_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    一个常见近似：只在采样到的 token 上比较 logprob 差
    return: scalar
    """
    token_kl = current_logprobs - ref_logprobs
    return masked_mean(token_kl, mask, dim=0)


# =========================
# 单步 GRPO loss
# =========================

def compute_grpo_loss(
    policy_model: AutoModelForCausalLM,
    old_model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    batch_data: Dict[str, List],
    advantages: torch.Tensor,   # [B, G]
    cfg: GRPOConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    返回：
    - 总 loss
    - 一些监控项
    """
    policy_model.train()
    old_model.eval()
    ref_model.eval()

    total_policy_loss = 0.0
    total_kl = 0.0
    total_tokens = 0

    B = len(batch_data["full_sequences"])
    G = len(batch_data["full_sequences"][0])

    for b in range(B):
        for g in range(G):
            seq = batch_data["full_sequences"][b][g]
            mask = batch_data["response_masks"][b][g]          # [seq_len-1]
            adv = advantages[b, g].detach()                    # 回答级 advantage

            current_logprobs = compute_sequence_token_logprobs(policy_model, seq)
            with torch.no_grad():
                old_logprobs = compute_sequence_token_logprobs(old_model, seq)
                ref_logprobs = compute_sequence_token_logprobs(ref_model, seq)

            ratio = torch.exp(current_logprobs - old_logprobs)  # [seq_len-1]
            clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)

            # 回答级 advantage 广播到 token 级
            adv_vec = torch.full_like(ratio, fill_value=adv.item())

            surrogate_1 = ratio * adv_vec
            surrogate_2 = clipped_ratio * adv_vec
            token_obj = torch.minimum(surrogate_1, surrogate_2)

            policy_obj = masked_mean(token_obj, mask, dim=0)
            policy_loss = -policy_obj

            kl = kl_from_logprobs(current_logprobs, ref_logprobs, mask)
            loss = policy_loss + cfg.kl_coef * kl

            response_token_count = int(mask.sum().item())
            total_policy_loss += policy_loss * response_token_count
            total_kl += kl * response_token_count
            total_tokens += response_token_count

    if total_tokens == 0:
        raise RuntimeError("No response tokens found. Check generation/masking.")

    mean_policy_loss = total_policy_loss / total_tokens
    mean_kl = total_kl / total_tokens
    total_loss = mean_policy_loss + cfg.kl_coef * mean_kl

    metrics = {
        "policy_loss": float(mean_policy_loss.detach().cpu().item()),
        "kl": float(mean_kl.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
    }
    return total_loss, metrics


# =========================
# 主训练循环
# =========================

def train_grpo(cfg: GRPOConfig) -> None:
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id

    policy_model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(cfg.device)

    # old_model: 每轮 rollout 前从当前 policy 拷一份冻结版本
    # ref_model: 长期冻结参考模型
    ref_model = copy.deepcopy(policy_model).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=cfg.lr)

    for step in range(cfg.train_steps):
        # 1) 采样 prompts
        prompt_batch = sample_prompts(PROMPTS, cfg.batch_size)

        # 2) old policy 冻结快照
        old_model = copy.deepcopy(policy_model).eval()
        for p in old_model.parameters():
            p.requires_grad = False

        # 3) 用 old policy 采样 group responses
        batch_data = generate_group_responses(
            model=old_model,
            tokenizer=tokenizer,
            prompts=prompt_batch,
            cfg=cfg,
        )

        # 4) reward
        rewards = compute_group_rewards(
            prompt_texts=batch_data["prompt_texts"],
            response_texts=batch_data["response_texts"],
            reward_fn=simple_reward_fn,
            device=cfg.device,
        )  # [B, G]

        # 5) 组内标准化 advantage
        advantages = group_normalize(rewards)  # [B, G]

        # 6) 计算 GRPO loss
        optimizer.zero_grad()
        loss, metrics = compute_grpo_loss(
            policy_model=policy_model,
            old_model=old_model,
            ref_model=ref_model,
            batch_data=batch_data,
            advantages=advantages,
            cfg=cfg,
        )
        loss.backward()
        optimizer.step()

        # 7) 打印
        reward_mean = rewards.mean().item()
        reward_std = rewards.std(unbiased=False).item()

        print(
            f"[step {step:03d}] "
            f"reward_mean={reward_mean:.4f} "
            f"reward_std={reward_std:.4f} "
            f"policy_loss={metrics['policy_loss']:.4f} "
            f"kl={metrics['kl']:.4f} "
            f"total_loss={metrics['total_loss']:.4f}"
        )

        # 顺便打印一组样本，方便观察
        if step % 10 == 0:
            print("Prompt:", batch_data["prompt_texts"][0])
            for gi in range(cfg.group_size):
                print(
                    f"  [{gi}] reward={rewards[0, gi].item():+.3f} "
                    f"adv={advantages[0, gi].item():+.3f} "
                    f"resp={repr(batch_data['response_texts'][0][gi])}"
                )
            print("-" * 80)


if __name__ == "__main__":
    cfg = GRPOConfig()
    train_grpo(cfg)