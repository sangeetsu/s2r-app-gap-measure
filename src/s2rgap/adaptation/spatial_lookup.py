# SPDX-License-Identifier: Apache-2.0
"""Memory-mapped lookup for precomputed Alpamayo spatial feature maps.

The extract step (``tools/extract_alpamayo_feature_maps.py``) saves a single
``.npz`` per dataset containing ``z_spatial`` (N, H, W, D) plus small metadata
arrays (paths, cells, hw). The .npz is stored *uncompressed*, so the big
``z_spatial.npy`` member can be unpacked verbatim into a standalone file that
NumPy can ``mmap_mode='r'``. That sidecar is created lazily on first load.

This avoids the alternative of loading the 11 GB tensor into a 31 GB-RAM host.

Usage::

    lookup = SpatialFeatureLookup.from_npz("~/datasets/kitti/spatial_features.npz")
    feat = lookup.get_batch(["frame_000123.png", "frame_001234.png"])
    # feat: torch.Tensor (B, H, W, D) fp16 on CPU.
"""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import torch


class SpatialFeatureLookup:
    """Path -> (H, W, D) feature-map lookup over a memory-mapped .npy."""

    def __init__(
        self,
        z_mmap: np.memmap,
        paths: list[str],
        target_hw: tuple[int, int],
    ) -> None:
        self.z_mmap = z_mmap  # (N, H, W, D) fp16, mmap'd
        self.paths = paths
        self.path_to_idx = {p: i for i, p in enumerate(paths)}
        self.target_hw = target_hw

    @classmethod
    def from_npz(cls, npz_path: str | Path) -> "SpatialFeatureLookup":
        npz_path = Path(npz_path).expanduser()
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)

        # The big tensor sidecar (mmap-friendly raw .npy).
        sidecar = npz_path.with_suffix(".z_spatial.npy")

        # Read small metadata directly from the .npz (cheap).
        with np.load(npz_path, allow_pickle=True) as blob:
            paths = [str(p) for p in blob["paths"].tolist()]
            target_hw_arr = blob["target_hw"]
            target_hw = (int(target_hw_arr[0]), int(target_hw_arr[1]))
            # Probe shape/dtype of z_spatial without loading it.
            # NpzFile lazily returns numpy arrays on access; we explicitly
            # avoid touching ``blob["z_spatial"]`` here so the 11 GB array
            # is never loaded into RAM.

        # Build / refresh the sidecar if missing.
        if not sidecar.exists():
            cls._extract_sidecar(npz_path, sidecar, member="z_spatial.npy")

        z_mmap = np.load(sidecar, mmap_mode="r")
        if z_mmap.shape[0] != len(paths):
            raise RuntimeError(
                f"sidecar N {z_mmap.shape[0]} != paths N {len(paths)} in {npz_path}"
            )
        return cls(z_mmap=z_mmap, paths=paths, target_hw=target_hw)

    @staticmethod
    def _extract_sidecar(npz_path: Path, sidecar: Path, member: str) -> None:
        """Stream a single .npy member out of an uncompressed .npz.

        zipfile.ZipFile streams the bytes to disk without loading the whole
        member into RAM, so this is safe on a 12-GB-free machine extracting
        an 11-GB member.
        """
        tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        with zipfile.ZipFile(npz_path) as zf:
            with zf.open(member) as src, open(tmp, "wb") as dst:
                # 4 MB chunks; zipfile streams uncompressed data verbatim.
                while True:
                    buf = src.read(4 * 1024 * 1024)
                    if not buf:
                        break
                    dst.write(buf)
        tmp.rename(sidecar)

    def __len__(self) -> int:
        return len(self.paths)

    def has(self, path: str) -> bool:
        return path in self.path_to_idx

    def get(self, path: str) -> torch.Tensor:
        """Return (H, W, D) fp16 tensor for a single frame."""
        idx = self.path_to_idx[path]
        return torch.from_numpy(np.array(self.z_mmap[idx]))

    def get_batch(self, paths: list[str]) -> torch.Tensor:
        """Return (B, H, W, D) fp16 tensor for a list of frames.

        Each lookup is a single mmap read into a fresh CPU buffer (np.array
        forces a copy off the memmap so the tensor is safe to hand to the
        GPU dataloader pipeline).
        """
        idxs = np.array([self.path_to_idx[p] for p in paths], dtype=np.int64)
        stacked = np.array(self.z_mmap[idxs])  # forces materialization
        return torch.from_numpy(stacked)
