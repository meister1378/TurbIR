"""Frame I/O for inference: a directory of frames or a video file -> uint8 grayscale stack, and back.

TurbIR takes 8-bit grayscale thermal frames. 16-bit radiometric frames are percentile-stretched
to 8 bit with one stretch for the whole sequence (a per-frame stretch would itself cause flicker).
"""
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    return img


def stretch_16bit(frames: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(frames, [low_pct, high_pct])
    hi = max(hi, lo + 1)
    return np.clip((frames.astype(np.float32) - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def read_frames(src: Path, max_frames: int = 0) -> tuple[np.ndarray, list[str]]:
    """Returns (T, H, W) uint8 and output file names. `src` is a frame directory (sorted by name) or a video."""
    src = Path(src)
    if src.is_dir():
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXT)
        if max_frames:
            files = files[:max_frames]
        if not files:
            raise SystemExit(f"no frames in {src}")
        frames = np.stack([to_gray(cv2.imread(str(f), cv2.IMREAD_UNCHANGED)) for f in files])
        names = [f.stem + ".png" for f in files]
    else:
        cap = cv2.VideoCapture(str(src))
        frames = []
        while not max_frames or len(frames) < max_frames:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(to_gray(img))
        cap.release()
        if not frames:
            raise SystemExit(f"could not read frames from {src}")
        frames = np.stack(frames)
        names = [f"{i:05d}.png" for i in range(len(frames))]
    if frames.dtype == np.uint16:
        frames = stretch_16bit(frames)
    elif frames.dtype != np.uint8:
        raise SystemExit(f"unsupported frame dtype {frames.dtype}")
    return frames, names


class VideoSink:
    """Optional mp4 output next to the PNG frames."""

    def __init__(self, path: Path, fps: float):
        self.path, self.fps, self.vw = Path(path), fps, None

    def write(self, img: np.ndarray):
        if self.vw is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = img.shape[:2]
            self.vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.vw.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

    def close(self):
        if self.vw is not None:
            self.vw.release()
