"""PyTorch retention index for an unmodified peptide.

The number is a learned estimate of the peptide's mean Kyte–Doolittle
hydrophobicity. On a reversed-phase column, a higher index means the peptide
is more likely to stick and elute later. It is not a measured retention time,
and it is not a mass.

The training labels are the mean of the published Kyte–Doolittle scale
(Kyte and Doolittle, 1982). The network sees only the sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from pep_mz.core.validation import validate_sequence

# Kyte–Doolittle hydrophobicity, J Mol Biol 157:105, 1982.
KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
_WEIGHTS = Path(__file__).with_name("retention_index.pt")


class RetentionNet(nn.Module):
    """Mean-pooled residue embedding, then a linear layer, then one index."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(len(AMINO_ACIDS), 4)
        self.head = nn.Linear(4, 1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(tokens) * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = hidden.sum(dim=1) / lengths
        return self.head(pooled).squeeze(-1)


def kyte_doolittle_mean(sequence: str) -> float:
    values = [KYTE_DOOLITTLE[aa] for aa in sequence]
    return sum(values) / len(values)


@dataclass(frozen=True)
class RetentionPrediction:
    sequence: str
    index: float
    """Predicted mean hydrophobicity. Higher sticks longer on a reversed-phase column."""
    reference_index: float
    """Kyte–Doolittle mean of the same sequence, for comparison."""


_MODEL: RetentionNet | None = None


def _load_model() -> RetentionNet:
    global _MODEL
    if _MODEL is None:
        model = RetentionNet()
        state = torch.load(_WEIGHTS, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _MODEL = model
    return _MODEL


def _encode(sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(seq) for seq in sequences)
    tokens = torch.zeros(len(sequences), width, dtype=torch.long)
    mask = torch.zeros(len(sequences), width, dtype=torch.float32)
    for row, seq in enumerate(sequences):
        for col, aa in enumerate(seq):
            tokens[row, col] = _AA_TO_INDEX[aa]
            mask[row, col] = 1.0
    return tokens, mask


def predict_retention(sequence: str) -> RetentionPrediction:
    """Score one unmodified peptide. Modifications are not part of the model."""
    validated = validate_sequence(sequence).sequence
    model = _load_model()
    tokens, mask = _encode([validated])
    with torch.no_grad():
        index = float(model(tokens, mask)[0])
    return RetentionPrediction(
        sequence=validated,
        index=index,
        reference_index=kyte_doolittle_mean(validated),
    )


def train_retention_model(path: Path | None = None, *, seed: int = 7) -> dict[str, float]:
    """Train on random peptides and write the weights. Returns validation error."""
    generator = torch.Generator().manual_seed(seed)
    amino = torch.tensor([_AA_TO_INDEX[aa] for aa in AMINO_ACIDS])

    def batch(n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = torch.randint(6, 26, (n,), generator=generator)
        width = int(lengths.max())
        tokens = torch.zeros(n, width, dtype=torch.long)
        mask = torch.zeros(n, width, dtype=torch.float32)
        target = torch.empty(n, dtype=torch.float32)
        scale = torch.tensor([KYTE_DOOLITTLE[aa] for aa in AMINO_ACIDS])
        draws = torch.randint(0, len(AMINO_ACIDS), (n, width), generator=generator)
        for i, length in enumerate(lengths.tolist()):
            chosen = amino[draws[i, :length]]
            tokens[i, :length] = chosen
            mask[i, :length] = 1.0
            target[i] = scale[draws[i, :length]].mean()
        return tokens, mask, target

    model = RetentionNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(120):
        tokens, mask, target = batch(1024, generator)
        optimizer.zero_grad()
        loss = loss_fn(model(tokens, mask), target)
        loss.backward()
        optimizer.step()

    model.eval()
    tokens, mask, target = batch(2000, generator)
    with torch.no_grad():
        pred = model(tokens, mask)
        mae = float((pred - target).abs().mean())
    torch.save(model.state_dict(), path or _WEIGHTS)
    global _MODEL
    _MODEL = None
    return {"validation_mae": mae}
