"""Optional TensorRT backend for the per-window step of the fast path (infer.py --trt).

The step is everything _FrameCache.run does after the per-frame cache: RAFT correlation + recurrent
updates for every (reference, frame) pair, SMD, feature/LR warps, LPCF, reconstruction and upsampling.
It is written out once more here with explicit dtypes and no Python control flow on data (the reference
frame is a one-hot input, so one engine serves every reference position), exported to ONNX and built as a
strongly typed TensorRT engine per (checkpoint, window length, tile shape). Engines are cached on disk.

dtypes: RAFT coordinates and all warps fp32; the correlation pyramid fp16 (the PyTorch path keeps it fp32,
but the looked-up values enter RAFT's fp16 update block either way, and the volume is the largest memory
traffic on big tiles); RAFT's update block fp16; SMD fp32; LPCF, reconstruction and upsampling fp16
(PyTorch uses bf16 there). The bicubic base is added in PyTorch. Not bit-identical to the PyTorch path;
validated by full-test PSNR (README).
"""
import copy
import hashlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TurbIR, dct_matrix


def _grid(h: int, w: int) -> torch.Tensor:
    gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32),
                            indexing="ij")
    return torch.stack((gx, gy), dim=-1)[None]  # (1, h, w, 2), x first


class TrtStep(nn.Module):
    """forward(lq (N,1,h,w) f32, fmap (N,C,fh,fw) f32, hidden (N,96,fh,fw) f32, context (N,64,fh,fw) f32,
    feat (N,64,h,w) f16, onehot (N,) f32) -> (detail (1,1,4h,4w) f32, lr_fused (1,1,h,w) f32).
    SR = bicubic_x4(lr_fused) + detail."""

    def __init__(self, model: TurbIR, n: int, h: int, w: int):
        super().__init__()
        fe, raft = model.flow, model.flow.raft
        assert raft.mask_predictor is None and model.smd is not None and model.lpcf is not None
        self.n, self.h, self.w = n, h, w
        self.hs, self.ws = max(h * fe.upscale, 128), max(w * fe.upscale, 128)
        self.fh, self.fw = (self.hs + (-self.hs) % 8) // 8, (self.ws + (-self.ws) % 8) // 8
        self.updates, self.levels, self.radius = fe.num_updates, raft.corr_block.num_levels, raft.corr_block.radius
        self.update_block = copy.deepcopy(raft.update_block).half()
        self.gate = copy.deepcopy(model.smd.gate).float()
        self.max_frames = model.smd.max_frames
        self.lpcf = copy.deepcopy(model.lpcf).half()
        self.recon = copy.deepcopy(model.recon).half()
        self.up = copy.deepcopy(model.up).half()
        r = self.radius
        d = torch.linspace(-r, r, 2 * r + 1)
        self.register_buffer("offsets", torch.stack(torch.meshgrid(d, d, indexing="ij"), dim=-1)[None])
        self.register_buffer("coords0", _grid(self.fh, self.fw).permute(0, 3, 1, 2).contiguous())  # (1,2,fh,fw)
        self.register_buffer("lr_grid", _grid(h, w))
        self.register_buffer("lr_scale", torch.tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)]))
        self.register_buffer("dct", dct_matrix(n))
        sizes, fh, fw = [], self.fh, self.fw
        for _ in range(self.levels):
            sizes.append((fh, fw))
            fh, fw = fh // 2, fw // 2
        self.register_buffer("lvl_scale", torch.tensor([[2.0 / (b - 1), 2.0 / (a - 1)] for a, b in sizes]))
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("lap", lap[None, None].half())

    def warp(self, x, flow):
        """model.flow_warp: x (B,C,h,w), flow (B,2,h,w) f32, border padding."""
        g = (self.lr_grid + flow.permute(0, 2, 3, 1)) * self.lr_scale - 1.0
        return F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)

    def flows(self, fmap, hidden, context, oh):
        n, c, fh, fw = fmap.shape
        hw = fh * fw
        oh4 = oh.view(n, 1, 1, 1)
        ref_t = ((fmap * oh4).sum(0, keepdim=True) / math.sqrt(c)).view(1, c, hw).transpose(1, 2)
        # RAFT pools the all-pairs volume over the target axes. That is linear, so pooling the target
        # feature map first and correlating each level directly gives the same values while never reading
        # the (n, hw, hw) volume again: far less memory traffic, which dominates on large tiles.
        pyramid, f, ref_h = [], fmap, ref_t.half()
        for lvl in range(self.levels):
            if lvl:
                f = F.avg_pool2d(f, kernel_size=2, stride=2)
            pyramid.append(torch.matmul(ref_h, f.half().reshape(n, c, -1)).view(n * hw, 1, f.shape[-2], f.shape[-1]))
        hid = (hidden * oh4).sum(0, keepdim=True).expand(n, -1, -1, -1).half()
        ctx = (context * oh4).sum(0, keepdim=True).expand(n, -1, -1, -1).half()
        coords1 = self.coords0.expand(n, -1, -1, -1)
        for _ in range(self.updates):
            cen = coords1.permute(0, 2, 3, 1).reshape(n * hw, 1, 1, 2)
            corr = torch.cat([F.grid_sample(v, ((cen / 2 ** lvl + self.offsets) * self.lvl_scale[lvl] - 1.0).half(),
                                            mode="bilinear", align_corners=True).view(n, fh, fw, -1)
                              for lvl, v in enumerate(pyramid)], dim=-1).permute(0, 3, 1, 2)
            hid, delta = self.update_block(hid, ctx, corr.half(), (coords1 - self.coords0).half())
            coords1 = coords1 + delta.float()
        flow = 8 * F.interpolate(coords1 - self.coords0, size=(fh * 8, fw * 8), mode="bilinear",
                                 align_corners=True)[..., :self.hs, :self.ws]
        flow = F.interpolate(flow, size=(self.h, self.w), mode="bilinear", align_corners=False)
        flow = flow * torch.tensor([self.w / self.ws, self.h / self.hs], device=flow.device).view(1, 2, 1, 1)
        return flow * (1.0 - oh4)  # ref -> ref is zero, as in TurbIR.compute_flows

    def smd(self, flows, err, oh):
        n, _, h, w = flows.shape
        coef = (self.dct @ flows.reshape(n, -1)).view(n, 2, h, w)
        mag = torch.sqrt((coef * coef).sum(1))[None]  # (1, n, h, w)
        mag = mag / (mag[:, 1:].mean(1, keepdim=True) + 1e-3)
        feat = torch.cat([mag, torch.zeros(1, self.max_frames - n, h, w, device=flows.device),
                          err.mean(0, keepdim=True)[:, None]], dim=1)
        gate = torch.sigmoid(self.gate(feat)[:, :n])
        gate = torch.cat([torch.ones_like(gate[:, :1]), gate[:, 1:]], dim=1)[0]  # (n, h, w)
        low = (self.dct.t() @ (coef * gate[:, None]).reshape(n, -1)).view(n, 2, h, w)
        low_ref = (low * oh.view(n, 1, 1, 1)).sum(0, keepdim=True)
        return flows - low_ref, flows - low  # warps, tilt

    def forward(self, lq, fmap, hidden, context, feat, onehot):
        n, _, h, w = lq.shape
        oh4 = onehot.view(n, 1, 1, 1)
        flows = self.flows(fmap, hidden, context, onehot)
        lq_ref = (lq * oh4).sum(0, keepdim=True)
        err = (lq_ref - self.warp(lq, flows)).abs()[:, 0]  # (n, h, w)
        warps, tilt = self.smd(flows, err, onehot)
        feats = self.warp(feat.float(), warps).half()  # (n, 64, h, w)
        lr_al = self.warp(lq, warps)  # (n, 1, h, w) f32
        tilt_mag = torch.sqrt((tilt * tilt).sum(1, keepdim=True))
        # LPCF (LuckyPatchFusion.forward) in fp16, softmax in fp32
        # lower median = the (n - (n-1)//2)-th largest. TopK over the last axis is exact and ~1 ms; TopK over
        # axis 0 is 14 ms and the min/max network of the PyTorch path becomes one extremely slow fused kernel
        lp, c = self.lpcf, feats.shape[1]
        med = feats.permute(1, 2, 3, 0).topk(n - (n - 1) // 2, dim=-1).values[..., -1][None]
        lr16 = lr_al.half()
        sharp = F.avg_pool2d(F.conv2d(F.pad(lr16, (1, 1, 1, 1), mode="replicate"), self.lap).abs(), 5, stride=1,
                             padding=2)
        inp = torch.cat([feats, feats - med, sharp, tilt_mag.half()], dim=1)
        score = lp.score(inp).float().view(n, lp.groups, 1, h, w)
        weights = torch.softmax(score, dim=0)
        fused = (weights.half() * feats.view(n, lp.groups, c // lp.groups, h, w)).sum(0).view(1, c, h, w)
        # temporal mean as a 1 x n GEMM: as a reduction TensorRT fuses it with the TopK transpose into a
        # pathological kernel (seconds per call)
        mean = (torch.full((1, n), 1.0 / n, device=feats.device, dtype=feats.dtype) @ feats.reshape(n, -1))
        out = lp.fuse(torch.cat([fused, mean.view(1, c, h, w)], dim=1))
        lr_fused = (weights.mean(1) * lr_al).sum(0, keepdim=True)  # (1, 1, h, w) f32
        x = out + self.recon(out)
        return self.up(x).float(), lr_fused


class TrtRunner:
    """Builds (or loads) the engine for one (checkpoint, n, h, w) and runs it on CUDA tensors."""

    def __init__(self, model: TurbIR, ckpt_path: str, n: int, h: int, w: int, cache_dir: str, log=print):
        import tensorrt as trt
        self.trt = trt
        sha = hashlib.sha1()
        for p in (ckpt_path, __file__):  # checkpoint + this graph definition
            with open(p, "rb") as f:
                sha.update(f.read())
        tag = sha.hexdigest()[:12]
        os.makedirs(cache_dir, exist_ok=True)
        stem = os.path.join(cache_dir, f"turbir_{tag}_n{n}_{h}x{w}_trt{trt.__version__}")
        logger = trt.Logger(trt.Logger.WARNING)
        if not os.path.exists(stem + ".engine"):
            log(f"building TensorRT engine for n={n}, {w}x{h} (once, cached in {cache_dir}) ...")
            step = TrtStep(model, n, h, w).cuda().eval()
            fh, fw = step.fh, step.fw
            ex = (torch.rand(n, 1, h, w, device="cuda"), torch.randn(n, 128, fh, fw, device="cuda"),
                  torch.randn(n, 96, fh, fw, device="cuda"), torch.randn(n, 64, fh, fw, device="cuda"),
                  torch.randn(n, 64, h, w, device="cuda").half(), F.one_hot(torch.tensor(n // 2), n).float().cuda())
            with torch.no_grad():
                torch.onnx.export(step, ex, stem + ".onnx", opset_version=17, dynamo=False,
                                  input_names=["lq", "fmap", "hidden", "context", "feat", "onehot"],
                                  output_names=["detail", "lr_fused"])
            builder = trt.Builder(logger)
            net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
            parser = trt.OnnxParser(net, logger)
            if not parser.parse_from_file(stem + ".onnx"):
                raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i))
                                                                   for i in range(parser.num_errors)))
            cfg = builder.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
            plan = builder.build_serialized_network(net, cfg)
            if plan is None:
                raise RuntimeError("TensorRT engine build failed")
            with open(stem + ".engine", "wb") as f:
                f.write(plan)
            os.remove(stem + ".onnx")
        with open(stem + ".engine", "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()  # TensorRT synchronises extra on the legacy default stream
        self.detail = torch.empty(1, 1, 4 * h, 4 * w, device="cuda")
        self.lr_fused = torch.empty(1, 1, h, w, device="cuda")
        self.ctx.set_tensor_address("detail", self.detail.data_ptr())
        self.ctx.set_tensor_address("lr_fused", self.lr_fused.data_ptr())

    def __call__(self, lq, fmap, hidden, context, feat, onehot):
        cur = torch.cuda.current_stream()
        for name, t in zip(["lq", "fmap", "hidden", "context", "feat", "onehot"],
                           (lq, fmap, hidden, context, feat, onehot)):
            t = t.contiguous()
            t.record_stream(self.stream)  # keep the memory alive until the engine has read it
            self.ctx.set_tensor_address(name, t.data_ptr())
        self.stream.wait_stream(cur)  # inputs ready; previous outputs consumed
        if not self.ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        cur.wait_stream(self.stream)
        return F.interpolate(self.lr_fused, scale_factor=4, mode="bicubic", align_corners=False) + self.detail
