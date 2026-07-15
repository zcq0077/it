"""Frozen-Qwen numeric candidate reranker for vessel trajectories."""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


DEFAULT_MARITIME_PROMPT = (
    "Rank vessel trajectory candidates by counterfactual reverse verification. For each candidate, "
    "ask whether that future route can explain the full observed history when traced backward along "
    "its shipping-lane prototype. Prefer monotonic route progress, decreasing cross-track error, "
    "heading and turn consistency, plausible reverse reconstruction, and agreement with available "
    "voyage context. Do not force a branch when the evidence is indistinguishable."
)


class QwenCandidateReranker(nn.Module):
    """Compare trajectory candidates with numeric soft tokens.

    The Qwen backbone is frozen. Only the numeric adapters, token-type
    embeddings, and score head are trained and saved.
    """

    def __init__(
            self,
            model_path,
            history_dim,
            candidate_dim,
            adapter_dim=256,
            gradient_checkpointing=True,
            dtype=torch.bfloat16,
            maritime_prompt=DEFAULT_MARITIME_PROMPT,
    ):
        super().__init__()
        self.model_path = str(model_path)
        self.history_dim = int(history_dim)
        self.candidate_dim = int(candidate_dim)
        self.adapter_dim = int(adapter_dim)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.maritime_prompt = str(maritime_prompt)

        self.backbone = AutoModel.from_pretrained(
            self.model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.backbone.config.use_cache = False
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if self.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        hidden_size = int(self.backbone.config.hidden_size)
        self.hidden_size = hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        prompt_ids = self.tokenizer(
            self.maritime_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        self.register_buffer("prompt_input_ids", prompt_ids, persistent=False)
        self.history_adapter = nn.Sequential(
            nn.LayerNorm(self.history_dim),
            nn.Linear(self.history_dim, self.adapter_dim),
            nn.GELU(),
            nn.Linear(self.adapter_dim, hidden_size),
        )
        self.candidate_adapter = nn.Sequential(
            nn.LayerNorm(self.candidate_dim),
            nn.Linear(self.candidate_dim, self.adapter_dim),
            nn.GELU(),
            nn.Linear(self.adapter_dim, hidden_size),
        )
        self.history_token_type = nn.Parameter(torch.zeros(hidden_size))
        self.candidate_token_type = nn.Parameter(torch.zeros(hidden_size))
        self.judge_token = nn.Parameter(torch.zeros(hidden_size))
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, self.adapter_dim),
            nn.GELU(),
            nn.Linear(self.adapter_dim, 1),
        )
        nn.init.normal_(self.history_token_type, std=0.02)
        nn.init.normal_(self.candidate_token_type, std=0.02)
        nn.init.normal_(self.judge_token, std=0.02)

        # These values are selected on the validation set after adapter training.
        self.fusion_weight = 0.0
        self.selector_margin_threshold = 0.0
        self.qwen_margin_threshold = 0.0

    @property
    def backbone_dtype(self):
        return next(self.backbone.parameters()).dtype

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def tokenize_contexts(self, contexts, max_length=64):
        encoded = self.tokenizer(
            list(contexts),
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max(int(max_length), 8),
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def forward(
            self,
            history,
            candidate_features,
            context_input_ids=None,
            context_attention_mask=None,
    ):
        if history.ndim != 3 or candidate_features.ndim != 3:
            raise ValueError("history and candidate_features must both be rank-3 tensors.")
        batch_size, candidate_count, _ = candidate_features.shape

        history_tokens = self.history_adapter(history.float()) + self.history_token_type
        candidate_tokens = self.candidate_adapter(candidate_features.float()) + self.candidate_token_type

        prompt_tokens = self.backbone.get_input_embeddings()(
            self.prompt_input_ids.to(history.device)
        ).expand(batch_size, -1, -1)
        judge_tokens = self.judge_token.view(1, 1, -1).expand(batch_size, 1, -1)
        embedding_parts = [prompt_tokens]
        mask_parts = [
            torch.ones(batch_size, prompt_tokens.size(1), device=history.device, dtype=torch.long)
        ]
        if context_input_ids is not None:
            context_input_ids = context_input_ids.to(history.device)
            context_tokens = self.backbone.get_input_embeddings()(context_input_ids)
            embedding_parts.append(context_tokens)
            if context_attention_mask is None:
                context_attention_mask = torch.ones_like(context_input_ids)
            mask_parts.append(context_attention_mask.to(history.device, dtype=torch.long))
        embedding_parts.extend((history_tokens, candidate_tokens, judge_tokens))
        mask_parts.extend((
            torch.ones(batch_size, history_tokens.size(1), device=history.device, dtype=torch.long),
            torch.ones(batch_size, candidate_tokens.size(1), device=history.device, dtype=torch.long),
            torch.ones(batch_size, 1, device=history.device, dtype=torch.long),
        ))
        inputs_embeds = torch.cat(embedding_parts, dim=1).to(self.backbone_dtype)
        attention_mask = torch.cat(mask_parts, dim=1)
        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        # The final judge token has attended to the maritime prompt, the full
        # observed history, and every candidate. Pair it with each candidate's
        # own numeric token so all candidates are compared in one Qwen pass.
        judge_hidden = outputs.last_hidden_state[:, -1, :].float()
        shared_judge = judge_hidden[:, None, :].expand(-1, candidate_count, -1)
        score_input = torch.cat((candidate_tokens.float(), shared_judge), dim=-1)
        return self.score_head(score_input).squeeze(-1)

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def adapter_state_dict(self):
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("backbone.")
        }

    def save_adapter(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 3,
                "model_path": self.model_path,
                "history_dim": self.history_dim,
                "candidate_dim": self.candidate_dim,
                "adapter_dim": self.adapter_dim,
                "gradient_checkpointing": self.gradient_checkpointing,
                "maritime_prompt": self.maritime_prompt,
                "state_dict": self.adapter_state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def from_adapter(cls, path, model_path=None, dtype=torch.bfloat16, map_location="cpu"):
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if int(payload.get("format_version", 1)) < 2:
            raise ValueError(
                "This adapter uses the legacy independent-candidate Qwen format. "
                "Retrain it with the current joint-candidate reranker."
            )
        reranker = cls(
            model_path=model_path or payload["model_path"],
            history_dim=payload["history_dim"],
            candidate_dim=payload["candidate_dim"],
            adapter_dim=payload["adapter_dim"],
            gradient_checkpointing=payload.get("gradient_checkpointing", True),
            dtype=dtype,
            maritime_prompt=payload.get("maritime_prompt", DEFAULT_MARITIME_PROMPT),
        )
        missing, unexpected = reranker.load_state_dict(payload["state_dict"], strict=False)
        missing = [name for name in missing if not name.startswith("backbone.")]
        if missing or unexpected:
            raise RuntimeError(
                f"Invalid Qwen reranker adapter: missing={missing}, unexpected={unexpected}"
            )
        metadata = payload.get("metadata", {})
        reranker.fusion_weight = float(metadata.get("fusion_weight", 0.0))
        reranker.selector_margin_threshold = float(
            metadata.get("selector_margin_threshold", 0.0)
        )
        reranker.qwen_margin_threshold = float(metadata.get("qwen_margin_threshold", 0.0))
        return reranker, metadata
