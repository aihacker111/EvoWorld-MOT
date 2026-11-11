"""Shared object-centric RSSM with a separate camera process and explicit kinematics.

Coordinate units: fractions of image width/height; dt is a number of video frames.
No metric-depth or absolute-physics claim is made for uncalibrated videos.
"""

from __future__ import annotations
from dataclasses import dataclass, fields
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from evoworld.config import Config
from evoworld.geometry import identity_affine, warp_boxes


def mlp(input_dim: int, hidden: int, output_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, output_dim)
    )


@dataclass
class ObjectState:
    h: Tensor
    z: Tensor
    box: Tensor
    velocity: Tensor
    valid: Tensor

    def detached(self):
        return ObjectState(**{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def clone(self):
        return ObjectState(**{f.name: getattr(self, f.name).clone() for f in fields(self)})

    def select(self, index):
        return ObjectState(**{f.name: getattr(self, f.name)[index] for f in fields(self)})


def damp_missing(state: ObjectState, observed: Tensor, delta: Tensor, decay: float) -> ObjectState:
    """Shared tracker rule (training and inference): unobserved velocity decays per frame.

    Open-loop constant velocity over long occlusions drifts predictions out of the scene.
    """
    if decay >= 1:
        return state
    factor = torch.as_tensor(decay, dtype=state.velocity.dtype, device=state.velocity.device)
    factor = factor ** delta.to(state.velocity).reshape(-1, 1, 1)
    missing = (state.valid & ~observed)[..., None]
    velocity = torch.where(missing, state.velocity * factor, state.velocity)
    return ObjectState(state.h, state.z, state.box, velocity, state.valid)


class NeighborInteraction(nn.Module):
    def __init__(self, hidden: int, neighbors: int):
        super().__init__()
        self.neighbors = neighbors
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.edge = mlp(4, hidden, hidden)
        self.out = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: Tensor, boxes: Tensor, valid: Tensor):
        B, N, D = h.shape
        if N == 0:
            return h
        # Distance matrix is O(N^2), attention gather is O(N*k); no hard target cap.
        distance = torch.cdist(boxes[..., :2].float(), boxes[..., :2].float())
        distance = distance.masked_fill(~valid[:, None, :], 1e6)
        indices = distance.topk(min(self.neighbors, N), largest=False).indices
        batch = torch.arange(B, device=h.device)[:, None, None]
        neighbor = h[batch, indices]
        rel = boxes[batch, indices] - boxes[:, :, None]
        keys = self.k(neighbor) + self.edge(rel)
        logits = (self.q(h)[:, :, None] * keys).sum(-1) / (D**0.5)
        mask = valid[batch, indices] & valid[:, :, None]
        logits = logits.masked_fill(~mask, -1e4)
        weight = logits.float().softmax(-1).to(h.dtype) * mask
        value = (self.v(neighbor) * weight[..., None]).sum(-2)
        return self.norm(h + self.out(value)) * valid[..., None]


class CameraDynamics(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.encoder = mlp(32, dimension, dimension)
        self.gru = nn.GRUCell(dimension, dimension)
        self.posterior_head = mlp(dimension, dimension, 6)
        self.prior_head = mlp(dimension, dimension, 6)
        for head in (self.posterior_head, self.prior_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def affine(parameters: Tensor):
        identity = identity_affine(parameters, parameters.shape[0])
        scale = parameters.new_tensor([[0.08, 0.08, 0.12], [0.08, 0.08, 0.12]])
        return identity + parameters.reshape(-1, 2, 3).tanh() * scale

    def observe(self, current: Tensor, previous: Tensor, state: Tensor | None):
        if state is None:
            state = current.new_zeros(current.shape[0], self.dimension)
        state = self.gru(self.encoder(torch.cat((current, previous), -1)), state)
        return state, self.affine(self.posterior_head(state))

    def imagine(self, state: Tensor):
        return self.affine(self.prior_head(state))


class ObjectRSSM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        H, Z, E = config.hidden_dim, config.stochastic_dim, config.embedding_dim
        self.obs = mlp(E + 4 + 1, H, H)
        self.interaction = NeighborInteraction(H, config.neighbors)
        self.gru = nn.GRUCell(Z + H + 6 + 1, H)
        self.prior = mlp(H, H, 2 * Z)
        self.posterior = mlp(2 * H, H, 2 * Z)
        self.birth = mlp(H, H, H)
        self.acceleration = mlp(H + Z, H, 2 * config.motion_modes)
        self.mode_logits = mlp(H + Z, H, config.motion_modes)
        self.size_rate = mlp(H + Z, H, 2)
        self.log_sigma = mlp(H + Z, H, 4)
        self.feature = mlp(H + Z, H, E)
        self.visibility = mlp(H + Z, H, 1)
        nn.init.zeros_(self.acceleration[-1].weight)
        nn.init.zeros_(self.acceleration[-1].bias)
        nn.init.zeros_(self.size_rate[-1].weight)
        nn.init.zeros_(self.size_rate[-1].bias)
        nn.init.zeros_(self.log_sigma[-1].weight)
        nn.init.constant_(self.log_sigma[-1].bias, -3.0)

    def initial(self, reference: Tensor, B: int, N: int):
        c = self.config
        return ObjectState(
            reference.new_zeros(B, N, c.hidden_dim),
            reference.new_zeros(B, N, c.stochastic_dim),
            reference.new_zeros(B, N, 4),
            reference.new_zeros(B, N, 2),
            torch.zeros(B, N, dtype=torch.bool, device=reference.device),
        )

    @staticmethod
    def distribution(parameters):
        mean, raw = parameters.float().chunk(2, -1)
        std = F.softplus(raw).clamp(0.1, 3.0)
        return mean, std

    @staticmethod
    def latent(mean, std, sample):
        return mean + std * torch.randn_like(mean) if sample else mean

    def predict(self, state: ObjectState, camera: Tensor, delta: Tensor, sample: bool = False):
        B, N, _ = state.h.shape
        interaction = self.interaction(state.h, state.box, state.valid)
        camera_input = camera.flatten(1)[:, None].expand(B, N, 6)
        dt = delta[:, None, None].expand(B, N, 1)
        h = self.gru(
            torch.cat((state.z, interaction, camera_input, dt), -1).reshape(B * N, -1),
            state.h.reshape(B * N, -1),
        ).reshape(B, N, -1)
        mean, std = self.distribution(self.prior(h))
        z = self.latent(mean, std, sample).to(h.dtype)
        latent = torch.cat((h, z), -1)
        accelerations = self.acceleration(latent).reshape(B, N, self.config.motion_modes, 2).tanh() * 0.03
        logits = self.mode_logits(latent)
        # Training has explicit mixture likelihood; recurrent rollout uses a persistent
        # sampled latent and a differentiable mean acceleration. Inference can inspect modes.
        acceleration = (accelerations * logits.softmax(-1)[..., None]).sum(-2)
        world = state.box.clone()
        world = torch.cat(
            (
                world[..., :2] + state.velocity * dt + 0.5 * acceleration * dt.square(),
                world[..., 2:].clamp_min(1e-4) * torch.exp(self.size_rate(latent).tanh() * 0.1 * dt),
            ),
            -1,
        )
        box = warp_boxes(world, camera)
        velocity = state.velocity + acceleration * dt
        velocity = torch.einsum("bij,bnj->bni", camera[..., :2], velocity)
        next_state = ObjectState(h, z, box, velocity, state.valid)
        centers = state.box[..., :2].unsqueeze(-2)
        centers = (
            centers
            + state.velocity.unsqueeze(-2) * dt.unsqueeze(-2)
            + 0.5 * accelerations * dt.unsqueeze(-2).square()
        )
        mode_boxes = torch.cat(
            (centers, world[..., 2:].unsqueeze(-2).expand(-1, -1, self.config.motion_modes, -1)), -1
        )
        mode_boxes = warp_boxes(mode_boxes, camera)
        info = dict(
            prior_mean=mean,
            prior_std=std,
            mode_boxes=mode_boxes,
            mode_logits=logits,
            log_sigma=self.log_sigma(latent).float().clamp(-6.0, -0.5),
            feature=F.normalize(self.feature(latent), dim=-1),
            visibility=self.visibility(latent).squeeze(-1),
        )
        return next_state, info

    def correct(
        self,
        predicted: ObjectState,
        box: Tensor,
        features: Tensor,
        observed: Tensor,
        delta: Tensor,
        previous_box: Tensor | None = None,
        sample: bool = False,
    ):
        observation = self.obs(torch.cat((features, box, observed[..., None].to(features)), -1))
        born = observed & ~predicted.valid
        h = torch.where(born[..., None], self.birth(observation), predicted.h)
        qmean, qstd = self.distribution(self.posterior(torch.cat((h, observation), -1)))
        pmean, pstd = self.distribution(self.prior(h))
        mean = torch.where(observed[..., None], qmean, pmean)
        std = torch.where(observed[..., None], qstd, pstd)
        z = self.latent(mean, std, sample).to(h.dtype)
        velocity = predicted.velocity
        if previous_box is not None:
            measured = (box[..., :2] - previous_box[..., :2]) / delta[:, None, None].clamp_min(1.0)
            velocity = torch.where((observed & ~born)[..., None], 0.5 * velocity + 0.5 * measured, velocity)
        corrected = ObjectState(
            h, z, torch.where(observed[..., None], box, predicted.box), velocity, predicted.valid | observed
        )
        # KL in FP32; only observed identities count, missing observations use prior.
        kl = (
            torch.log(pstd / qstd) + (qstd.square() + (qmean - pmean).square()) / (2 * pstd.square()) - 0.5
        ).sum(-1)
        return corrected, kl * observed
