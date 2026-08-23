"""The text latent space: PCA fitted inside a single fold, on train text only.

This is the one component that is *fitted on data*, so it is the one place
leakage can enter. It is therefore constructed from training indices only and
never sees a test embedding — not even to compute the mean.
"""

from __future__ import annotations

import numpy as np


class TextLatent:
    """384-d sentence embedding  <->  k-d latent, via PCA on the training fold."""

    def __init__(self, train_emb: np.ndarray, max_dim: int):
        """
        train_emb : (n_train, 384) embeddings of the TRAINING samples only.
        max_dim   : requested latent size; capped by the rank of the data.
        """
        # Deduplicate first: with few unique strings, repeated rows would let
        # frequency dominate the principal directions.
        uniq = np.unique(train_emb, axis=0)
        rank_cap = max(1, len(uniq) - 1)
        self.dim = int(min(max_dim, rank_cap, train_emb.shape[1]))

        self.mean = uniq.mean(axis=0, keepdims=True).astype(np.float32)
        centred = uniq - self.mean
        # economy SVD; rows of Vt are the principal directions
        _, sv, vt = np.linalg.svd(centred, full_matrices=False)
        self.basis = vt[: self.dim].astype(np.float32)          # (k, 384)

        total = float((sv ** 2).sum()) or 1.0
        self.explained = float((sv[: self.dim] ** 2).sum() / total)

        # scale so latents are roughly unit-variance — flow matching behaves
        # much better when the data and the noise live on the same scale
        codes = centred @ self.basis.T
        self.scale = np.maximum(codes.std(axis=0, keepdims=True), 1e-6).astype(np.float32)

    def encode(self, emb: np.ndarray) -> np.ndarray:
        """(n, 384) -> (n, k)"""
        return ((emb - self.mean) @ self.basis.T) / self.scale

    def decode(self, code: np.ndarray) -> np.ndarray:
        """(n, k) -> (n, 384), renormalised to the unit sphere like the encoder's output."""
        out = (code * self.scale) @ self.basis + self.mean
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norm, 1e-8)

    def reconstruct(self, emb: np.ndarray) -> np.ndarray:
        """Round-trip through the latent space — the ceiling any model can reach."""
        return self.decode(self.encode(emb))
