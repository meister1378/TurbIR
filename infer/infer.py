"""TurbIR inference: turbulent low-resolution thermal frames -> x4 restored frames.

Every output frame t is restored from its own window of `num_frames` input frames centred on t (shifted
inward at the start and end of the sequence). Frames larger than
`tile` are restored in overlapping spatial tiles with a feathered blend, because RAFT's all-pairs
correlation volume grows with (H*W)^2.
"""
import functools
import math
import os
import queue
import threading
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow.raft import make_coords_grid, upsample_flow

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "model"))
sys.path.insert(0, _HERE)
from ckpt_crypto import load_checkpoint  # noqa: E402
from data_io import VideoSink, read_frames  # noqa: E402
from model import TurbIR, flow_warp, local_sharpness  # noqa: E402

_CKPT_DIR = os.path.join(_HERE, "..", "checkpoints")
SCALE = 4


@functools.lru_cache(maxsize=None)
def _median_network(n: int) -> list[tuple[int, int]]:
    """Comparators (a, b) of Batcher's odd-even merge sort on the next power of two >= n, pruned to the
    ones that can reach the lower-median slot (n - 1) // 2. Slots >= n hold +inf."""
    size = 1 << (n - 1).bit_length()
    pairs, p = [], 1
    while p < size:
        k = p
        while k >= 1:
            for j in range(k % p, size - k, 2 * k):
                for i in range(min(k, size - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    need, keep = {(n - 1) // 2}, []
    for a, b in reversed(pairs):
        if a in need or b in need:
            keep.append((a, b))
            need |= {a, b}
    return keep[::-1]


def _median(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """== x.median(dim, keepdim=True).values (bit-identical: a selection network only moves values), but
    elementwise min/max on contiguous slices instead of a strided sort: 9.4 -> 2.8 ms for 15 x 64 x 192 x 256."""
    v = list(x.unbind(dim))
    n = len(v)
    v += [None] * ((1 << (n - 1).bit_length()) - n)  # None = +inf
    for a, b in _median_network(n):
        if v[b] is None:
            continue
        if v[a] is None:
            v[a], v[b] = v[b], None
        else:
            v[a], v[b] = torch.minimum(v[a], v[b]), torch.maximum(v[a], v[b])
    return v[(n - 1) // 2].unsqueeze(dim)


class _FrameCache:
    """Streaming form of TurbIR.forward for sliding-window inference.

    TurbIR.forward(window, ref) recomputes everything for all N frames of every window, although
    consecutive windows share N-1 frames. Per-frame quantities that do not depend on the reference frame
    are computed once per frame (and per tile) and reused:
      - the pixel feature extractor self.feat(frame),
      - RAFT's feature-encoder map of the prepared frame (instance norm: per image),
      - RAFT's context encoding of the frame (needed when it is the reference).
    RAFT then runs only the correlation + recurrent updates for the N-1 (ref, frame) pairs (the ref->ref
    flow is zeroed by the model anyway), and upsamples only the last update, the one the model uses.
    raft_dtype=torch.float16 runs RAFT's recurrent update block in fp16 (same 10-bit mantissa as the TF32
    convolutions it replaces; encoders, correlation and flow arithmetic stay fp32).
    The correlation pyramid and its lookup are torchvision's CorrBlock with the 1/sqrt(C) scale folded into
    the reference feature map and the coordinate normalisation precomputed; the LPCF temporal median uses a
    selection network (_median). Everything after the flow is TurbIR.forward verbatim.

    Outputs are not bit-identical to the plain forward: RAFT's TF32 convolutions already change by up to
    ~1 px of flow when only the batch size changes, and caching changes batch sizes. Measured on the whole
    FLIR-IVSR test split, PSNR against ground truth is unchanged (see README).
    """

    def __init__(self, model: TurbIR, amp: bool, raft_dtype=None, trt=None):
        """trt: callable (n, h, w) -> trt_engine.TrtRunner; the per-window step then runs in TensorRT."""
        self.m, self.amp, self.raft_dtype, self.trt = model, amp, raft_dtype, trt
        self.store = {}  # (tile key, frame index) -> dict

    def drop_before(self, first: int):
        for k in [k for k in self.store if k[1] < first]:
            del self.store[k]

    def _fill(self, key, idx, lq_tiles):
        """lq_tiles: (K, 1, h, w) for the frame indices `idx` that are not cached yet."""
        fe, raft = self.m.flow, self.m.flow.raft
        h, w = lq_tiles.shape[-2:]
        s = fe.upscale
        hs, ws = max(h * s, 128), max(w * s, 128)
        ph, pw = (-hs) % 8, (-ws) % 8
        with torch.autocast(lq_tiles.device.type, enabled=False):
            x = F.interpolate(lq_tiles.float(), size=(hs, ws), mode="bicubic", align_corners=False)
            x = F.pad(x.expand(-1, 3, -1, -1) * 2.0 - 1.0, (0, pw, 0, ph), mode="replicate")
            fmap = raft.feature_encoder(x)
            ctx = raft.context_encoder(x)
            hsz = raft.update_block.hidden_state_size
            hidden, context = torch.split(ctx, [hsz, ctx.shape[1] - hsz], dim=1)
            hidden, context = torch.tanh(hidden), F.relu(context)
        with torch.autocast(lq_tiles.device.type, dtype=torch.bfloat16, enabled=self.amp):
            feat = self.m.feat(lq_tiles)
        if self.trt is not None:
            feat = feat.half()  # exact: bf16 values are representable in fp16
        for j, i in enumerate(idx):
            self.store[(key, i)] = dict(lq=lq_tiles[j], fmap=fmap[j], hidden=hidden[j], context=context[j],
                                        feat=feat[j])

    def _flows(self, ent, ref):
        """(1, N, 2, h, w) flows ref -> every frame, as TurbIR.compute_flows."""
        fe, raft = self.m.flow, self.m.flow.raft
        n = len(ent)
        h, w = ent[0]["lq"].shape[-2:]
        hs, ws = max(h * fe.upscale, 128), max(w * fe.upscale, 128)
        others = [j for j in range(n) if j != ref]
        b = len(others)
        with torch.autocast(ent[0]["lq"].device.type, enabled=False):
            fmap2 = torch.stack([ent[j]["fmap"] for j in others])
            c, fh, fw = fmap2.shape[1:]
            # CorrBlock.build_pyramid: (fmap1^T fmap2) / sqrt(C), then avg-pooled num_levels - 1 times
            fmap1 = (ent[ref]["fmap"] / math.sqrt(c)).view(1, c, fh * fw).transpose(1, 2).expand(b, -1, -1)
            vol = torch.bmm(fmap1, fmap2.view(b, c, fh * fw)).view(b * fh * fw, 1, fh, fw)
            pyramid = [vol]
            for _ in range(raft.corr_block.num_levels - 1):
                pyramid.append(F.avg_pool2d(pyramid[-1], kernel_size=2, stride=2))
            # CorrBlock.index_pyramid with torchvision's grid_sample normalisation precomputed per level
            r = raft.corr_block.radius
            d = torch.linspace(-r, r, 2 * r + 1, device=fmap2.device)
            offsets = torch.stack(torch.meshgrid(d, d, indexing="ij"), dim=-1)[None]
            norm = [torch.tensor([2.0 / (v.shape[-1] - 1), 2.0 / (v.shape[-2] - 1)], device=fmap2.device)
                    for v in pyramid]

            def lookup(coords):
                cen = coords.permute(0, 2, 3, 1).reshape(b * fh * fw, 1, 1, 2)
                out = [F.grid_sample(v, (cen / 2 ** lvl + offsets) * nm - 1, mode="bilinear", align_corners=True)
                       .view(b, fh, fw, -1) for lvl, (v, nm) in enumerate(zip(pyramid, norm))]
                return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous()

            hidden = ent[ref]["hidden"][None].expand(b, -1, -1, -1)
            context = ent[ref]["context"][None].expand(b, -1, -1, -1)
            coords0 = make_coords_grid(b, fh, fw).to(fmap2.device)
            coords1 = make_coords_grid(b, fh, fw).to(fmap2.device)
            for _ in range(fe.num_updates):
                corr = lookup(coords1)
                with torch.autocast(fmap2.device.type, dtype=self.raft_dtype or torch.float16,
                                    enabled=self.raft_dtype is not None):
                    hidden, delta = raft.update_block(hidden, context, corr, coords1 - coords0)
                hidden, delta = hidden.float(), delta.float()
                coords1 = coords1 + delta
            up_mask = None if raft.mask_predictor is None else raft.mask_predictor(hidden)
            flow = upsample_flow(flow=coords1 - coords0, up_mask=up_mask)[..., :hs, :ws]
            flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
            flow = flow * torch.tensor([w / ws, h / hs], device=flow.device, dtype=flow.dtype).view(1, 2, 1, 1)
            flows = torch.zeros(n, 2, h, w, device=flow.device, dtype=flow.dtype)
            flows[others] = flow
        return flows[None]

    def _lpcf(self, feats, lr_aligned, tilt_mag):
        """LuckyPatchFusion.forward (first two outputs) with the temporal median from _median."""
        lp = self.m.lpcf
        b, n, c, h, w = feats.shape
        med = _median(feats, 1).expand_as(feats)
        sharp = local_sharpness(lr_aligned.flatten(0, 1)).view(b, n, 1, h, w)
        inp = torch.cat([feats, feats - med, sharp, tilt_mag], dim=2).flatten(0, 1)
        weights = torch.softmax(lp.score(inp).view(b, n, lp.groups, 1, h, w), dim=1)
        fused = (weights * feats.view(b, n, lp.groups, c // lp.groups, h, w)).sum(1).view(b, c, h, w)
        out = lp.fuse(torch.cat([fused, feats.mean(1)], dim=1))
        lr_fused = (weights.mean(2) * lr_aligned).sum(1)
        return out, lr_fused

    def run(self, key, lq_window: torch.Tensor, first: int, ref: int) -> torch.Tensor:
        """lq_window (n, 1, h, w) = frames first..first+n-1 of this tile -> SR of frame first+ref."""
        n = lq_window.shape[0]
        miss = [j for j in range(n) if (key, first + j) not in self.store]
        if miss:
            self._fill(key, [first + j for j in miss], lq_window[miss])
        ent = [self.store[(key, first + j)] for j in range(n)]
        if self.trt is not None:
            h, w = lq_window.shape[-2:]
            stack = lambda k: torch.stack([e[k] for e in ent])  # noqa: E731
            onehot = torch.eye(n, device=lq_window.device)[ref]
            return self.trt(n, h, w)(lq_window, stack("fmap"), stack("hidden"), stack("context"), stack("feat"),
                                     onehot)
        m = self.m
        with torch.autocast(lq_window.device.type, dtype=torch.bfloat16, enabled=self.amp):
            flows = self._flows(ent, ref)
            _, _, _, h, w = flows.shape
            lq = lq_window[None]
            lq_flat = lq_window
            err = (lq[:, ref:ref + 1]
                   - flow_warp(lq_flat, flows.flatten(0, 1)).view(1, n, 1, h, w)).abs().squeeze(2)
            if m.smd is not None:
                warps, _, tilt = m.smd(flows, ref, err)
            else:
                warps, tilt = flows, flows
            warps_flat = warps.flatten(0, 1)
            feats = flow_warp(torch.stack([e["feat"] for e in ent]), warps_flat).view(1, n, -1, h, w)
            lr_aligned = flow_warp(lq_flat, warps_flat).view(1, n, 1, h, w)
            if m.lpcf is not None:
                fused, lr_fused = self._lpcf(feats, lr_aligned, tilt.norm(dim=2, keepdim=True))
            else:
                fused, lr_fused = feats.mean(1), lr_aligned.mean(1)
            x = fused + m.recon(fused)
            base = F.interpolate(lr_fused, scale_factor=4, mode="bicubic", align_corners=False)
            sr = base + m.up(x)
        return sr.float()


def _fit_tile(size: int, tile: int, overlap: int) -> int:
    """Smallest tile (multiple of 4) that covers `size` with as few tiles as `tile` would, at `overlap`.
    E.g. 480 with tile 264 / overlap 24 -> 2 tiles of 252 instead of 2 of 264: the work per tile grows
    faster than its area (RAFT's correlation volume is quadratic in it), so no tile is larger than needed."""
    if size <= tile:
        return size
    k = math.ceil((size - overlap) / (tile - overlap))
    return min(tile, size, 4 * math.ceil((size + overlap * (k - 1)) / (4 * k)))


def _starts(size: int, tile: int, step: int) -> list[int]:
    """Fewest tile origins covering `size` with overlap >= tile - step, spread evenly."""
    if size <= tile:
        return [0]
    n = math.ceil((size - tile) / step) + 1
    return [round(i * (size - tile) / (n - 1)) for i in range(n)]


class TurbIRRestorer:
    """Loads TurbIR once; restore_sequence(frames) yields restored uint8 frames one by one.

    num_frames: temporal window (default 15). tile: (H, W) LR tile, None = never tile.
    precision: "bf16" (default, as evaluated) or "fp32"; the RAFT flow always runs in fp32.
    fast: reuse per-frame computations across sliding windows (_FrameCache); False runs the plain
    TurbIR.forward on every window (reference path, slower). raft_fp16: RAFT update block in fp16 (fast path).
    trt: run the per-window step of the fast path as a TensorRT engine (trt_engine.py; needs the tensorrt
    package; one engine per window length and tile shape is built on first use and cached in trt_cache_dir).
    """

    def __init__(self, ckpt_path, device="cuda", num_frames=15, tile=(264, 344), overlap=24, precision="bf16",
                 fast=True, raft_fp16=True, trt=False, trt_cache_dir=None):
        self.device = torch.device(device)
        self.model = TurbIR(pretrained_flow=False).to(self.device).eval()
        sd = load_checkpoint(ckpt_path)
        self.model.load_state_dict(sd)
        self.num_frames = num_frames
        self.tile = tile
        self.overlap = overlap
        self.amp = precision == "bf16" and self.device.type == "cuda"
        self.fast = fast or trt
        self.raft_dtype = torch.float16 if raft_fp16 and self.device.type == "cuda" else None
        self._trt = None
        if trt:
            if self.device.type != "cuda":
                raise ValueError("TensorRT needs a CUDA device")
            from trt_engine import TrtRunner
            cache_dir = trt_cache_dir or os.path.join(_HERE, "..", "trt_cache")
            runners = {}

            def get(n, h, w):
                if (n, h, w) not in runners:
                    runners[(n, h, w)] = TrtRunner(self.model, ckpt_path, n, h, w, cache_dir)
                return runners[(n, h, w)]

            self._trt = get

    @torch.no_grad()
    def _run(self, window: torch.Tensor, ref: int) -> torch.Tensor:
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            sr, _ = self.model(window, ref)
        return sr.float()

    @torch.no_grad()
    def _run_tiled(self, window: torch.Tensor, ref: int, run=None) -> torch.Tensor:
        """run(y, x, th, tw) -> SR of that LR tile; default: plain forward on the cropped window."""
        if run is None:
            run = lambda y, x, th, tw: self._run(window[..., y:y + th, x:x + tw], ref)  # noqa: E731
        _, _, _, h, w = window.shape
        th, tw = (_fit_tile(h, self.tile[0], self.overlap), _fit_tile(w, self.tile[1], self.overlap)) \
            if self.tile else (h, w)
        if th == h and tw == w:
            return run(0, 0, h, w)
        ov, s = self.overlap, SCALE
        out = torch.zeros(1, 1, h * s, w * s, device=window.device)
        acc = torch.zeros_like(out)
        ramp = lambda n: torch.clamp((torch.arange(n, device=window.device) + 0.5) / (ov * s), max=1.0)  # noqa: E731
        for y in _starts(h, th, th - ov):
            for x in _starts(w, tw, tw - ov):
                sr = run(y, x, th, tw)
                ry, rx = ramp(th * s), ramp(tw * s)
                wy = torch.minimum(ry, ry.flip(0)) if ov else torch.ones_like(ry)
                wx = torch.minimum(rx, rx.flip(0)) if ov else torch.ones_like(rx)
                if y == 0:  # frame borders get full weight (no neighbour to blend with)
                    wy[: th * s // 2] = 1
                if y + th == h:
                    wy[th * s // 2:] = 1
                if x == 0:
                    wx[: tw * s // 2] = 1
                if x + tw == w:
                    wx[tw * s // 2:] = 1
                wgt = wy[:, None] * wx[None, :]
                ys, xs = slice(y * s, (y + th) * s), slice(x * s, (x + tw) * s)
                out[..., ys, xs] += sr * wgt
                acc[..., ys, xs] += wgt
        return out / acc

    @torch.no_grad()
    def restore_sequence(self, frames: np.ndarray):
        """frames: (T, H, W) uint8 -> yields (t, (4H, 4W) uint8)."""
        lq = torch.from_numpy(np.ascontiguousarray(frames)).float().div(255).to(self.device)
        t_total, h, w = lq.shape
        n = min(self.num_frames, t_total)
        cache = _FrameCache(self.model, self.amp, self.raft_dtype, self._trt) if self.fast else None
        cuda = self.device.type == "cuda"
        pending = None  # (t, pinned host buffer, copy-done event) of the previous frame
        for r in range(t_total):
            s = min(max(r - n // 2, 0), t_total - n)
            window = lq[s:s + n].view(1, n, 1, h, w)
            if cache is None:
                sr = self._run_tiled(window, r - s)
            else:
                cache.drop_before(s)
                sr = self._run_tiled(window, r - s, run=lambda y, x, th, tw: cache.run(
                    (y, x), lq[s:s + n, None, y:y + th, x:x + tw], s, r - s))
            out = (sr.clamp(0, 1) * 255).round().byte().squeeze()
            if not cuda:
                yield r, out.numpy()
                continue
            # copy asynchronously and hand out frame r only after frame r + 1 is queued, so the GPU never
            # waits for the caller (PNG encoding etc.)
            host = torch.empty(out.shape, dtype=torch.uint8, pin_memory=True)
            host.copy_(out, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
            if pending is not None:
                pending[2].synchronize()
                yield pending[0], pending[1].numpy().copy()
            pending = (r, host, done)
        if pending is not None:
            pending[2].synchronize()
            yield pending[0], pending[1].numpy().copy()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Restore a turbulent low-resolution thermal sequence (x4).")
    p.add_argument("--input", required=True, help="directory of consecutive frames, or a video file")
    p.add_argument("--out-dir", required=True, help="restored PNG frames are written here")
    p.add_argument("--ckpt", default=os.path.join(_CKPT_DIR, "turbir_p.pth.enc"))
    p.add_argument("--num-frames", type=int, default=15, help="temporal window per output frame")
    p.add_argument("--tile", type=int, nargs=2, default=[264, 344], metavar=("H", "W"),
                   help="maximum LR tile size for large frames (default fits 12 GB); tiles are shrunk to just "
                        "cover the frame; 0 0 disables tiling")
    p.add_argument("--tile-overlap", type=int, default=24)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--video", help="also write an mp4 of the restored frames")
    p.add_argument("--fps", type=float, default=10.0, help="fps of --video")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-fast", action="store_true",
                   help="run the plain model forward on every window instead of reusing per-frame work")
    p.add_argument("--raft-fp32", action="store_true", help="keep RAFT's update block in fp32 (fast path)")
    p.add_argument("--trt", action="store_true",
                   help="run the per-window step as a TensorRT engine (built once per frame/tile shape, cached)")
    p.add_argument("--trt-cache", help="engine cache directory (default: ../trt_cache)")
    a = p.parse_args()

    frames, names = read_frames(Path(a.input), a.max_frames)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tile = tuple(a.tile) if all(a.tile) else None
    restorer = TurbIRRestorer(a.ckpt, a.device, a.num_frames, tile, a.tile_overlap, a.precision,
                              fast=not a.no_fast, raft_fp16=not a.raft_fp32, trt=a.trt, trt_cache_dir=a.trt_cache)
    sink = VideoSink(Path(a.video), a.fps) if a.video else None
    todo, errors = queue.Queue(maxsize=8), []

    def writer():  # PNG / video encoding off the main thread (cv2 releases the GIL)
        while (item := todo.get()) is not None:
            if errors:
                continue  # keep draining so the main thread never blocks
            try:
                cv2.imwrite(str(out_dir / names[item[0]]), item[1])
                if sink:
                    sink.write(item[1])
            except Exception as e:  # noqa: BLE001 - re-raised below
                errors.append(e)

    th = threading.Thread(target=writer, daemon=True)
    th.start()
    t0 = time.time()
    for item in restorer.restore_sequence(frames):
        todo.put(item)
    todo.put(None)
    th.join()
    if errors:
        raise errors[0]
    if sink:
        sink.close()
    dt = time.time() - t0
    print(f"{len(frames)} frames {frames.shape[2]}x{frames.shape[1]} -> x4 in {dt:.1f} s "
          f"({len(frames) / dt:.2f} fps) -> {out_dir}")
