"""Compact dense ReID head for the JDE model.

The head receives the complete stride-4 P2 feature and emits a dense embedding
map.  The optional concentration head is used only while training vMFMarginID.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardJDEReID(nn.Module):
    """Conventional shared-P2 JDE ReID head."""

    def __init__(
        self,
        c: int,
        reid_dim: int = 128,
        use_kappa: bool = False,
        kappa_init: float = 10.0,
        kappa_max: float = 30.0,
    ):
        super().__init__()
        self.c = int(c)
        self.reid_dim = int(reid_dim)
        self.use_kappa = bool(use_kappa)
        self.kappa_max = float(kappa_max)
        self.reid_head = nn.Sequential(
            nn.Conv2d(self.c, self.c, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.c),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c, self.reid_dim, 1, bias=True),
        )

        for m in self.reid_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.use_kappa:
            self.kappa_head = nn.Conv2d(self.reid_dim, 1, 1)
            nn.init.zeros_(self.kappa_head.weight)
            kappa_init = max(float(kappa_init), 1e-4)
            bias = math.log(math.expm1(kappa_init)) if kappa_init < 20.0 else kappa_init
            nn.init.constant_(self.kappa_head.bias, bias)
        else:
            self.kappa_head = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb_map = self.reid_head(x)
        kappa_map = None
        if self.training and self.kappa_head is not None:
            kappa_map = F.softplus(self.kappa_head(emb_map)).clamp(max=self.kappa_max)
        return emb_map, kappa_map
