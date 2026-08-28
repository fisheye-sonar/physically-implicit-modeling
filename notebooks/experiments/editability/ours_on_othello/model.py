"""OUR transformer world model, adapted to Othello's input/output scheme.

The mirror of `othello_transfer/`. That thread runs **our probe and our editor on their
model**; this one trains **our architecture, with our recipe, on their world**. Together
with the two results already in hand they complete a 2x2:

| | their world | our world (discworld) |
|---|---|---|
| **their architecture** | editable (2026-08-20/21) | `directions/othello-architecture-on-discworld.md` — not run |
| **our architecture** | **this thread** | not editable (2026-08-04 … 2026-08-21) |

Exactly three things change from `pim.world_models.transformer.TransformerModel`:

    encoder   Linear(128, d_model) + ReLU  ->  nn.Embedding(61, d_model)
    decoder   Linear(d_model, 128)         ->  Linear(d_model, 61)
    loss      MSE on the next observation  ->  cross-entropy on the next move

Everything else is the discworld model untouched: RoPE, pre-norm blocks, `d_model` 256,
4 layers, 4 heads, mlp_ratio 4, banded-causal attention. Sevan's call (2026-08-21) was to
replace the **whole** encoder — the `Linear` *and* the `ReLU` — since a ReLU on an embedding
lookup would zero half the dimensions for no reason.

Why the band width is not a problem
-----------------------------------
The obvious objection is that a window-16 model cannot see a 60-move game. It can:
stacking widens the receptive field, so

    state_span = n_layers * (window - 1) + 1 = 4 * 15 + 1 = 61

which covers a full game exactly. (At `window` 4 the span is 13 and this test would be
void — the fit at our canonical `window` 16 is luck worth stating.) What the band *does*
cost is directness: their model attends from the last move to any earlier move in one hop
at all 8 layers, while ours must route through up to 4 banded hops. `window` 40 (span 157)
is run alongside as the arm that removes that difference, at **zero** extra compute — the
band mask is applied to the full T x T attention, so all widths cost the same (measured:
25,018 vs 25,001 games/s).

Interface
---------
`othello_transfer`'s probe and intervention code calls seven names on a model. Our model
already has all of them natively, including `_run`'s `fn(layer, x) -> x` edit hook — so
unlike `othello_shim.OthelloGPTShim`, **there is no shim here**. Only `embed`,
`residual_stack` and `decode` are overridden, to take `(B, T)` move tokens instead of an
observation buffer. Residual point `ell` is the stream after `ell` blocks, `n_layers + 1`
points, the same convention as everywhere else in the repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pim.world_models.transformer.model import ModelConfig, TransformerModel

VOCAB = 61  # pad + the 60 non-centre squares; `othello_data.canonical_vocab()`
BLOCK = 59  # their `block_size`: the model never sees more than 59 moves


class OthelloTransformer(TransformerModel):
    """`TransformerModel` with a token embedding in and move logits out."""

    def __init__(self, cfg: ModelConfig, vocab: int = VOCAB) -> None:
        super().__init__(cfg)
        self.vocab = vocab
        self.encoder = nn.Embedding(vocab, cfg.d_model)
        self.decoder = nn.Linear(cfg.d_model, vocab)
        self.state_view = "activations"  # the only view that means anything here

    # ── the three overrides ───────────────────────────────────────────────────

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) move tokens -> (B, T, d_model). Residual point 0.

        No ReLU: the discworld encoder is `relu(Linear(obs))`, but an embedding lookup is
        already a free parameter per token, so a ReLU would only zero half of it.
        """
        return self.encoder(idx)

    def logits(self, idx: torch.Tensor, edit=None) -> torch.Tensor:
        """(B, T, vocab) next-move logits at **every** position — the training view."""
        h, _ = self._run(self.embed(idx), self._seq_mask(idx.shape[1], idx.device), edit=edit)
        return self.decoder(self.norm_out(h))

    def decode(self, idx: torch.Tensor, edit=None) -> torch.Tensor:
        """(B, vocab) next-move logits at the **last** position — the intervention view.

        Signature-identical to `othello_shim.OthelloGPTShim.decode`, so
        `linear_intervention.run`, `transfer_pipeline.run_arm` and
        `othello_probe.make_intervention_hook` drive this model unchanged.
        """
        h, _ = self._run(self.embed(idx), self._seq_mask(idx.shape[1], idx.device), edit=edit)
        return self.decoder(self.norm_out(h[:, -1]))

    @torch.no_grad()
    def residual_stack(self, idx: torch.Tensor, edit=None) -> torch.Tensor:
        """(n_layers+1, B, T, d_model) — the stream at every residual point."""
        _, resids = self._run(
            self.embed(idx), self._seq_mask(idx.shape[1], idx.device), edit=edit, want_resid=True
        )
        return torch.stack(resids, 0)

    # ── deliberately absent ───────────────────────────────────────────────────

    def advance(self, state, obs_t):
        raise NotImplementedError(
            "Rolling forward in Othello needs design decisions this thread has not made "
            "(does the model's own move enter the history? sampled or argmax? how does the "
            "counterfactual board evolve alongside?). Like the Li replication, every "
            "measurement here is step-0."
        )

    def predict_step(self, state):
        raise NotImplementedError(self.advance.__doc__)


def build(d_model: int = 256, n_layers: int = 4, n_heads: int = 4, window: int = 16,
          mlp_ratio: float = 4.0, vocab: int = VOCAB) -> OthelloTransformer:
    """Defaults are `W16`'s, i.e. the discworld transformer's, verbatim."""
    cfg = ModelConfig(input_dim=128, d_model=d_model, n_layers=n_layers,
                      n_heads=n_heads, mlp_ratio=mlp_ratio, window=window)
    return OthelloTransformer(cfg, vocab=vocab)
