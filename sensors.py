import numpy as np

import config as cfg


def cast(pos, angle, fld):
    ang = angle[:, None] + cfg.RAY_ANGLES[None, :]
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    steps = (np.arange(cfg.RAY_SAMPLES) + 1.0) * cfg.RAY_STEP
    probes = pos[:, None, None, :] + dirs[:, :, None, :] * steps[None, None, :, None]

    beyond_wall = fld.sample(probes) < 0.0
    hit = beyond_wall.any(axis=2)
    first = np.argmax(beyond_wall, axis=2)
    return np.where(hit, (first + 1.0) * cfg.RAY_STEP, cfg.RAY_MAX)
