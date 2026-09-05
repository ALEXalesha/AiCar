import os

import numpy as np

import config as cfg
import dataset
import track

MODEL_FILE = cfg.MODEL_FILE


def available(path=MODEL_FILE):
    return os.path.exists(path)


class Generator:
    def __init__(self, path=MODEL_FILE):
        with np.load(path) as data:
            self.latent = int(data["latent"][0])
            self.n_points = int(data["n_points"][0])
            keys = sorted((k for k in data.files if k.startswith("w")), key=lambda k: int(k[1:]))
            self.layers = [(data[k].astype(float), data["b" + k[1:]].astype(float)) for k in keys]

    def decode(self, z):
        h = np.atleast_2d(z)
        last = len(self.layers) - 1
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            h = 1.0 / (1.0 + np.exp(-h)) if i == last else np.maximum(h, 0.0)
        return h

    def sample_radii(self, rng, temperature=1.0):
        z = rng.normal(0.0, temperature, self.latent)
        return dataset.denormalise(self.decode(z)[0])

    def make_track(self, rng, width=None, difficulty=None, attempts=40, temperature=1.0):
        width = cfg.TRACK_WIDTH if width is None else width
        difficulty = cfg.DIFFICULTY if difficulty is None else difficulty

        best, best_score = None, -1.0
        for _ in range(attempts):
            center = track.centerline_from_radii(self.sample_radii(rng, temperature))
            if not track.walls_are_sane(center, width, difficulty):
                continue
            if not (cfg.LEN_MIN <= track.polyline_length(center) <= cfg.LEN_MAX):
                continue
            score = track.interest(center)
            if score > best_score:
                best, best_score = center, score
        return track.Track(best if best is not None else track.circle_centerline(), width)
