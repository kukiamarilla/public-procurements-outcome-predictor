"""Embedder por chunks a partir de un causal LM (Hugging Face), congelado."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lm_config import ModelConfig


def _backbone_causal_lm(lm: nn.Module) -> nn.Module | None:
    """Submódulo que devuelve `last_hidden_state` (Llama: .model, GPT-2: .transformer)."""
    for attr in ("model", "transformer"):
        sub = getattr(lm, attr, None)
        if isinstance(sub, nn.Module):
            return sub
    return None


class ChunkEmbedder(nn.Module):
    """
    Chunk embedder sobre un causal LM (parámetros congelados).

    Entrada: text (str)
    Salida: Tensor [N, d_in], N = cantidad de chunks.
    """

    def __init__(
        self,
        gpt_model: nn.Module,
        tokenizer,
        max_len: int = 4096,
        stride: int = 2048,
        device: str = "cuda",
        chunk_batch_size: int = 4,
    ):
        super().__init__()
        self.gpt = gpt_model.eval()
        for param in self.gpt.parameters():
            param.requires_grad_(False)

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.device = device
        self.chunk_batch_size = max(1, int(chunk_batch_size))

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id

        self._backbone = _backbone_causal_lm(self.gpt)

    def _forward_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids, attention_mask: [B, L]
        Devuelve última capa oculta [B, L, H] sin materializar todas las capas (cuando hay backbone).
        """
        if self._backbone is not None:
            outputs = self._backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            return outputs.last_hidden_state
        outputs = self.gpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    @torch.no_grad()
    def forward(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )

        input_ids = encoded["input_ids"][0].to(self.device)
        attention_mask = encoded["attention_mask"][0].to(self.device)
        total_len = int(attention_mask.sum().item())

        chunk_ids: list[torch.Tensor] = []
        chunk_masks: list[torch.Tensor] = []

        for start in range(0, max(1, total_len), self.stride):
            end = min(start + self.max_len, total_len)

            ids = input_ids[start:end]
            mask = attention_mask[start:end]

            if ids.numel() == 0:
                continue

            pad_len = self.max_len - ids.numel()
            if pad_len > 0:
                ids = torch.cat(
                    [
                        ids,
                        torch.full(
                            (pad_len,),
                            self.pad_id,
                            device=self.device,
                            dtype=ids.dtype,
                        ),
                    ],
                    dim=0,
                )
                mask = torch.cat(
                    [
                        mask,
                        torch.zeros(
                            (pad_len,),
                            device=self.device,
                            dtype=mask.dtype,
                        ),
                    ],
                    dim=0,
                )

            chunk_ids.append(ids)
            chunk_masks.append(mask)

            if end >= total_len:
                break

        if not chunk_ids:
            raise ValueError("No chunks could be generated from the provided text.")

        chunk_embeddings: list[torch.Tensor] = []
        B = self.chunk_batch_size
        for i in range(0, len(chunk_ids), B):
            batch_ids = torch.stack(chunk_ids[i : i + B], dim=0)
            batch_mask = torch.stack(chunk_masks[i : i + B], dim=0)
            hidden = self._forward_hidden(batch_ids, batch_mask)
            for j in range(hidden.shape[0]):
                last_valid_idx = int(batch_mask[j].sum().item()) - 1
                chunk_embeddings.append(hidden[j, last_valid_idx].detach().clone())

        return torch.stack(chunk_embeddings, dim=0)


def build_chunk_embedder(config: ModelConfig) -> ChunkEmbedder:
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dev = torch.device(config.device)
    raw = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=config.dtype,
    )
    base_model = cast(nn.Module, raw).to(dev)

    return ChunkEmbedder(
        gpt_model=base_model,
        tokenizer=tokenizer,
        max_len=config.max_len,
        stride=config.stride,
        device=config.device,
        chunk_batch_size=config.chunk_batch_size,
    )


def infer_input_dim(embedder: ChunkEmbedder) -> int:
    mc = embedder.gpt.config
    hs = getattr(mc, "hidden_size", None)
    if hs is not None:
        return int(hs)
    ne = getattr(mc, "n_embd", None)
    if ne is not None:
        return int(ne)
    raise ValueError("Could not infer the embedding dimension from the base model config.")
