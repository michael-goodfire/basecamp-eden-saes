"""EDEN-7B partial-forward harness.

Loads an EDEN-7B checkpoint (e.g. EDEN-7B-BCR) frozen in bf16 and runs only the
embeddings + decoder layers ``0..L`` (via goodfire-core's ``early_stop`` hook),
capturing the layer-``L`` residual stream and skipping layers ``L+1..`` and the
LM head. Shared by both activation backends (pre-harvest and on-the-fly).

Token ids are the byte-level ids produced by
:mod:`basecamp_eden_saes.corpus.build` (raw uppercase A/C/G/T = bytes
65/67/71/84). EDEN-BCR's tokenizer has no auto-prepended special token
(``post_processor: null``); ``prefix_token`` is exposed only so the
elevated-output-head id-2 hypothesis can be tested empirically. Default is no
prefix.
"""

from __future__ import annotations

import numpy as np
import torch

from goodfire_core.models.hf_transformers import HfTransformerModel
from goodfire_core.models.interfaces import HookSpec

PAD_ID = 49  # EDEN tokenizer pad token ("1")
EOS_ID = 48  # EDEN tokenizer eos token ("0")


class EdenPartialForward:
    """Frozen EDEN-7B forward truncated at a chosen decoder layer."""

    def __init__(
        self,
        model_path: str,
        layer: int = 24,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq: int = 4096,
        prefix_token: int | None = None,
        pad_id: int = PAD_ID,
    ):
        self.model = HfTransformerModel(
            model_path, device=device, dtype=dtype, no_grad=True
        )
        self.layer = layer
        self.site = f"model.layers.{layer}"
        self.device = device
        self.dtype = dtype
        self.max_seq = max_seq
        self.prefix_token = prefix_token
        self.pad_id = pad_id
        self.d_model = self._infer_d_model()
        # ensure frozen
        for p in self.model.model.parameters():
            p.requires_grad_(False)
        self.model.model.eval()

    def _infer_d_model(self) -> int:
        for attr in ("d_hidden", "d_model"):
            v = getattr(self.model, attr, None)
            if isinstance(v, int) and v > 0:
                return v
        return int(self.model.model.config.hidden_size)

    def build_batch(
        self, windows: list[np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad token windows into ``(input_ids, attn_mask, keep_mask)``.

        Right-padding + a causal mask means real-token activations are identical
        to the unpadded forward. ``keep_mask`` excludes pads (and the optional
        prefix token) so only genuine sequence positions are returned downstream.
        """
        pre = 1 if self.prefix_token is not None else 0
        lengths = [len(w) + pre for w in windows]
        maxlen = max(lengths)
        if maxlen > self.max_seq:
            raise ValueError(
                f"window+prefix length {maxlen} exceeds max_seq {self.max_seq}"
            )
        b = len(windows)
        ids = np.full((b, maxlen), self.pad_id, dtype=np.int64)
        attn = np.zeros((b, maxlen), dtype=np.int64)
        keep = np.zeros((b, maxlen), dtype=bool)
        for i, w in enumerate(windows):
            if pre:
                ids[i, 0] = self.prefix_token
            ids[i, pre : pre + len(w)] = w
            attn[i, : pre + len(w)] = 1
            keep[i, pre : pre + len(w)] = True
        return (
            torch.from_numpy(ids).to(self.device),
            torch.from_numpy(attn).to(self.device),
            torch.from_numpy(keep).to(self.device),
        )

    @torch.no_grad()
    def residual(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run embeddings + layers 0..L, return layer-L residual ``[B, S, d]``."""
        if input_ids.shape[1] > self.max_seq:
            raise ValueError(
                f"seq len {input_ids.shape[1]} exceeds max_seq {self.max_seq}"
            )
        spec = HookSpec(sites=[self.site], early_stop=True)
        _, capture = self.model.forward_with_hooks(
            input_ids, spec, attention_mask=attention_mask
        )
        return capture.acts[self.site]

    @torch.no_grad()
    def tokens_from_windows(self, windows: list[np.ndarray]) -> torch.Tensor:
        """Partial-forward a batch of windows; return valid-token acts ``[N, d]``."""
        ids, attn, keep = self.build_batch(windows)
        acts = self.residual(ids, attn)  # [B, S, d]
        return acts[keep]  # [N_valid, d]

    @torch.no_grad()
    def residual_multi(
        self,
        input_ids: torch.Tensor,
        layers: list[int],
        attention_mask: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        """Run embeddings + layers 0..max(layers), return per-layer residuals.

        Captures every requested decoder layer's residual stream in a single
        forward pass. ``early_stop`` halts after the deepest requested layer, so
        layers beyond max(layers) and the LM head are never computed -- the
        cost of capturing {16, 24, 28} is one partial forward to layer 28.
        """
        if input_ids.shape[1] > self.max_seq:
            raise ValueError(
                f"seq len {input_ids.shape[1]} exceeds max_seq {self.max_seq}"
            )
        ordered = sorted(set(int(x) for x in layers))
        sites = [f"model.layers.{ell}" for ell in ordered]
        spec = HookSpec(sites=sites, early_stop=True)
        _, capture = self.model.forward_with_hooks(
            input_ids, spec, attention_mask=attention_mask
        )
        return {ell: capture.acts[f"model.layers.{ell}"] for ell in ordered}

    @torch.no_grad()
    def tokens_from_windows_multi(
        self, windows: list[np.ndarray], layers: list[int]
    ) -> dict[int, torch.Tensor]:
        """Partial-forward a batch of windows; return per-layer valid-token acts.

        Returns ``{layer: [N_valid, d]}`` with the pad/prefix positions removed,
        identical token selection to :meth:`tokens_from_windows` but for several
        layers at once.
        """
        ids, attn, keep = self.build_batch(windows)
        acts = self.residual_multi(ids, layers, attn)  # {layer: [B, S, d]}
        return {ell: a[keep] for ell, a in acts.items()}
