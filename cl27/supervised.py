"""Raw-CLS cross-entropy classifier shared by supervised control methods."""

from __future__ import annotations

from torch import nn


class CEClassifier(nn.Module):
    """A single native backbone and linear CE head, without feature normalization."""

    def __init__(self, backbone, head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.feature_dim = backbone.out_dim

    def extract_vector(self, images):
        return self.backbone(images)

    def forward(self, images):
        if self.head is None:
            raise RuntimeError("The CE classifier has no registered classes yet")
        return {"logits": self.head(self.backbone(images))}
