"""TurbIR: turbulence-aware multi-frame thermal video super-resolution (x4).

One window of N turbulent low-resolution frames -> the x4 restored reference frame, in a single pass:

    RAFT-small flow (reference -> every frame)
    -> SMD: temporal DCT of each pixel's flow trajectory + learned per-pixel gate splits real motion
       (temporal low-pass) from turbulence tilt; de-tilt warp W_t = F_t - L_ref
    -> warp per-frame features (and LR frames) into the tilt-free reference geometry
    -> LPCF: lucky-imaging style per-(frame, pixel, channel-group) confidence, softmax over time
    -> 20 ResBlocks + PixelShuffle x2 x2, added to the bicubic x4 of the confidence-fused LR frame.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small


# --------------------------------------------------------------------------------------------- flow

def flow_warp(x: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """out(p) = x(p + flow(p)); flow is (B, 2, H, W) in pixels, channel 0 = dx."""
    b, _, h, w = x.shape
    # The pixel grid is always fp32: bf16 holds integers exactly only up to 256, so a grid in x.dtype
    # would sample bf16 inputs up to 1 px off beyond row/column 256.
    gy, gx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=torch.float32),
        torch.arange(w, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    px = gx + flow[:, 0]
    py = gy + flow[:, 1]
    grid = torch.stack((2.0 * px / max(w - 1, 1) - 1.0, 2.0 * py / max(h - 1, 1) - 1.0), dim=-1)
    return F.grid_sample(x, grid, mode=mode, padding_mode="border", align_corners=True)


class FlowEstimator(nn.Module):
    """RAFT-small (torchvision) on gray frames; forward(ref, tgt) returns F with ref(p) ~= tgt(p + F(p)).

    Frames are upsampled x`upscale` before RAFT: turbulence tilt is often sub-pixel at LR, and RAFT needs
    inputs of at least 128 px. Use `pretrained=False` with a TurbIR checkpoint, which holds every weight.
    """

    def __init__(self, num_updates: int = 6, upscale: int = 2, pretrained: bool = True):
        super().__init__()
        self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT if pretrained else None, progress=False)
        self.num_updates = num_updates
        self.upscale = upscale

    def forward(self, ref: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        h, w = ref.shape[-2:]
        s = self.upscale
        hs, ws = max(h * s, 128), max(w * s, 128)
        ph, pw = (-hs) % 8, (-ws) % 8

        def prep(x):
            x = F.interpolate(x, size=(hs, ws), mode="bicubic", align_corners=False)
            return F.pad(x.expand(-1, 3, -1, -1) * 2.0 - 1.0, (0, pw, 0, ph), mode="replicate")

        flow = self.raft(prep(ref), prep(tgt), num_flow_updates=self.num_updates)[-1][..., :hs, :ws]
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        return flow * torch.tensor([w / ws, h / hs], device=flow.device, dtype=flow.dtype).view(1, 2, 1, 1)


# ---------------------------------------------------------------------- SMD (motion / tilt split)

def dct_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II matrix D (n x n): coeffs = D @ signal, signal = D.T @ coeffs."""
    k = torch.arange(n, dtype=torch.float64)[:, None]
    t = torch.arange(n, dtype=torch.float64)[None, :]
    d = torch.cos(math.pi * (t + 0.5) * k / n) * math.sqrt(2.0 / n)
    d[0] /= math.sqrt(2.0)
    return d.to(device=device, dtype=dtype)


class SpectralMotionDecomposition(nn.Module):
    """Splits reference->frame flow trajectories into motion (temporal low-pass) and turbulence tilt.

    Frame t shows the scene displaced by smooth motion plus a zero-mean (over time) tilt. With flows F_t,
    the temporal low-pass L of F holds motion and the reference's own tilt (DC); the warp that maps frame t
    onto the tilt-free geometry at the reference time is W_t = F_t - L_ref. Which DCT bins count as "low"
    is a per-pixel learned gate.
    """

    def __init__(self, max_frames: int = 32, hidden: int = 32, dc_only: bool = False):
        super().__init__()
        self.max_frames = max_frames
        self.dc_only = dc_only  # temporal mean only
        self.gate = nn.Sequential(
            nn.Conv2d(max_frames + 1, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, max_frames, 3, padding=1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.linspace(4.0, -4.0, max_frames))

    def forward(self, flows: torch.Tensor, ref_idx: int, err: torch.Tensor):
        """flows: (B, N, 2, H, W); err: (B, N, H, W) photometric error after warping with F.
        Returns de-tilt warps (B, N, 2, H, W), gate (B, N, H, W), tilt (B, N, 2, H, W)."""
        b, n, _, h, w = flows.shape
        d = dct_matrix(n, flows.device, flows.dtype)
        coef = torch.einsum("kt,btchw->bkchw", d, flows)
        mag = coef.norm(dim=2)
        mag = mag / (mag[:, 1:].mean(1, keepdim=True) + 1e-3)
        feat = torch.zeros(b, self.max_frames, h, w, device=flows.device, dtype=flows.dtype)
        feat[:, :n] = mag
        feat = torch.cat([feat, err.mean(1, keepdim=True)], dim=1)
        gate = torch.sigmoid(self.gate(feat)[:, :n])
        if self.dc_only:
            gate = torch.zeros_like(gate)
        gate = torch.cat([torch.ones_like(gate[:, :1]), gate[:, 1:]], dim=1)  # DC is always low-pass
        low = torch.einsum("kt,bkchw->btchw", d, coef * gate[:, :, None])
        return flows - low[:, ref_idx:ref_idx + 1], gate, flows - low


# ------------------------------------------------------------------------- LPCF (temporal fusion)

def local_sharpness(x: torch.Tensor) -> torch.Tensor:
    """Local Laplacian energy (B*N, 1, H, W), a proxy for momentary blur."""
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device, dtype=x.dtype)
    lap = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k[None, None])
    return F.avg_pool2d(lap.abs(), 5, stride=1, padding=2)


class LuckyPatchFusion(nn.Module):
    """Per (frame, pixel, channel-group) confidence from the aligned feature, its difference to the temporal
    median, the local sharpness of the aligned LR frame and the residual tilt magnitude; softmax over time."""

    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        self.groups = groups
        self.score = nn.Sequential(
            nn.Conv2d(2 * channels + 2, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, groups, 3, padding=1),
        )
        self.fuse = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, feats: torch.Tensor, lr_aligned: torch.Tensor, tilt_mag: torch.Tensor):
        """feats (B, N, C, H, W); lr_aligned (B, N, 1, H, W); tilt_mag (B, N, 1, H, W)."""
        b, n, c, h, w = feats.shape
        med = feats.median(dim=1, keepdim=True).values.expand_as(feats)
        sharp = local_sharpness(lr_aligned.flatten(0, 1)).view(b, n, 1, h, w)
        inp = torch.cat([feats, feats - med, sharp, tilt_mag], dim=2).flatten(0, 1)
        weights = torch.softmax(self.score(inp).view(b, n, self.groups, 1, h, w), dim=1)
        fused = (weights * feats.view(b, n, self.groups, c // self.groups, h, w)).sum(1).view(b, c, h, w)
        out = self.fuse(torch.cat([fused, feats.mean(1)], dim=1))
        lr_fused = (weights.mean(2) * lr_aligned).sum(1)
        return out, lr_fused, weights.mean(2).squeeze(2)


# ---------------------------------------------------------------------------------------- TurbIR

class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
                                  nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class TurbIR(nn.Module):
    """forward(lq (B, N, 1, h, w) in [0, 1], ref_idx) -> (sr (B, 1, 4h, 4w), aux dict)."""

    def __init__(self, channels: int = 64, n_feat_blocks: int = 5, n_recon_blocks: int = 20,
                 use_smd: bool = True, use_lpcf: bool = True, max_frames: int = 32, flow_updates: int = 6,
                 smd_dc_only: bool = False, pretrained_flow: bool = True):
        super().__init__()
        c = channels
        self.flow = FlowEstimator(num_updates=flow_updates, pretrained=pretrained_flow)
        self.smd = SpectralMotionDecomposition(max_frames, dc_only=smd_dc_only) if use_smd else None
        self.feat = nn.Sequential(nn.Conv2d(1, c, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
                                  *[ResBlock(c) for _ in range(n_feat_blocks)])
        self.lpcf = LuckyPatchFusion(c) if use_lpcf else None
        self.recon = nn.Sequential(*[ResBlock(c) for _ in range(n_recon_blocks)], nn.Conv2d(c, c, 3, padding=1))
        self.up = nn.Sequential(
            nn.Conv2d(c, 4 * c, 3, padding=1), nn.PixelShuffle(2), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, 4 * c, 3, padding=1), nn.PixelShuffle(2), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, 1, 3, padding=1),
        )

    def compute_flows(self, lq: torch.Tensor, ref_idx: int) -> torch.Tensor:
        b, n, _, h, w = lq.shape
        ref = lq[:, ref_idx:ref_idx + 1].expand(-1, n, -1, -1, -1).flatten(0, 1)
        with torch.autocast(lq.device.type, enabled=False):  # RAFT in fp32
            flows = self.flow(ref.float(), lq.flatten(0, 1).float()).view(b, n, 2, h, w)
        mask = torch.ones(1, n, 1, 1, 1, device=lq.device)
        mask[:, ref_idx] = 0
        return flows * mask

    def forward(self, lq: torch.Tensor, ref_idx: int):
        b, n, _, h, w = lq.shape
        flows = self.compute_flows(lq, ref_idx)
        lq_flat = lq.flatten(0, 1)
        err = (lq[:, ref_idx:ref_idx + 1]
               - flow_warp(lq_flat, flows.flatten(0, 1)).view(b, n, 1, h, w)).abs().squeeze(2)
        if self.smd is not None:
            warps, gate, tilt = self.smd(flows, ref_idx, err)
        else:
            warps, gate, tilt = flows, None, flows
        warps_flat = warps.flatten(0, 1)
        feats = flow_warp(self.feat(lq_flat), warps_flat).view(b, n, -1, h, w)
        lr_aligned = flow_warp(lq_flat, warps_flat).view(b, n, 1, h, w)
        if self.lpcf is not None:
            fused, lr_fused, weights = self.lpcf(feats, lr_aligned, tilt.norm(dim=2, keepdim=True))
        else:
            fused, lr_fused, weights = feats.mean(1), lr_aligned.mean(1), None
        x = fused + self.recon(fused)
        base = F.interpolate(lr_fused, scale_factor=4, mode="bicubic", align_corners=False)
        return base + self.up(x), {"flows": flows, "warps": warps, "gate": gate, "weights": weights}


def build_model(cfg: dict | None = None, pretrained_flow: bool = True) -> TurbIR:
    """TurbIR from a config dict; None -> released config."""
    cfg = cfg or {}
    return TurbIR(channels=cfg.get("channels", 64), n_recon_blocks=cfg.get("recon_blocks", 20),
                  use_smd=not cfg.get("no_smd", False), use_lpcf=not cfg.get("no_lpcf", False),
                  flow_updates=cfg.get("flow_updates", 6), smd_dc_only=cfg.get("smd_dc_only", False),
                  pretrained_flow=pretrained_flow)
