"""Single owner of all trainable modules and portable checkpoints."""

from torch import nn
from evoworld.config import Config
from .perception import build_perception
from .world import ObjectRSSM, CameraDynamics
from .association import Association
from .life import LifeController


class EvoWorld(nn.Module):
    def __init__(self, config: Config, initialize: bool = True):
        super().__init__()
        self.config = config.validate()
        self.perception = build_perception(config, initialize)
        self.world = ObjectRSSM(config)
        self.camera = CameraDynamics(config.camera_dim)
        self.association = Association(config)
        if config.tracker_mode == "al":
            self.al = LifeController(config)

    def parameter_report(self):
        return {name: sum(p.numel() for p in module.parameters()) for name, module in self.named_children()}
