import numpy as np

import config as cfg

OUTSIDE = -1e6


class Field:
    def __init__(self, values, origin, cell):
        self.values = values
        self.origin = origin
        self.cell = cell

    @property
    def shape(self):
        return self.values.shape

    def sample(self, pts):
        idx = np.floor((pts - self.origin) / self.cell).astype(np.int64)
        ix, iy = idx[..., 0], idx[..., 1]
        nx, ny = self.values.shape
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        return np.where(inside, self.values[np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)], OUTSIDE)


def build(center, half_width, cell=cfg.FIELD_CELL, pad=cfg.FIELD_PAD):
    lo = center.min(axis=0) - half_width - pad
    hi = center.max(axis=0) + half_width + pad
    nx = int(np.ceil((hi[0] - lo[0]) / cell)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / cell)) + 1

    gx = lo[0] + (np.arange(nx) + 0.5) * cell
    gy = lo[1] + (np.arange(ny) + 0.5) * cell
    grid = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)

    dist = np.empty(len(grid))
    for i in range(0, len(grid), cfg.FIELD_CHUNK):
        chunk = grid[i:i + cfg.FIELD_CHUNK]
        diff = chunk[:, None, :] - center[None, :, :]
        dist[i:i + cfg.FIELD_CHUNK] = np.sqrt(np.einsum("cpk,cpk->cp", diff, diff)).min(axis=1)

    return Field((half_width - dist).reshape(nx, ny), lo, cell)


def build_for_track(trk, cell=cfg.FIELD_CELL, pad=cfg.FIELD_PAD):
    return build(trk.center, trk.half, cell, pad)
