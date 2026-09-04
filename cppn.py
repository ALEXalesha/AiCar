import numpy as np

import config as cfg


def _gauss(x):
    return np.exp(-x * x)


ACTS = (np.sin, _gauss, np.tanh, np.abs)


def genome_size(layers):
    return sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))


def _unpack(genome, layers):
    need = genome_size(layers)
    if len(genome) != need:
        raise ValueError(f"геном длины {len(genome)}, для слоёв {layers} нужно {need}")
    ws, bs, k = [], [], 0
    for i in range(len(layers) - 1):
        n_in, n_out = layers[i], layers[i + 1]
        ws.append(genome[k:k + n_in * n_out].reshape(n_in, n_out))
        k += n_in * n_out
        bs.append(genome[k:k + n_out])
        k += n_out
    return ws, bs


def _mixed(h):
    out = np.empty_like(h)
    for j in range(h.shape[1]):
        out[:, j] = ACTS[j % len(ACTS)](h[:, j])
    return out


def forward(genome, layers, x):
    ws, bs = _unpack(genome, layers)
    h = x
    last = len(ws) - 1
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w + b
        h = np.tanh(h) if i == last else _mixed(h)
    return h


def ring_shape(genome, layers, n_points, r_min, r_max):
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    x = np.stack([np.sin(theta), np.cos(theta)], axis=1)
    r = forward(genome, layers, x)[:, 0]
    r = r_min + (r + 1.0) * 0.5 * (r_max - r_min)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def random_genome(layers, rng, scale=cfg.CPPN_INIT_SCALE):
    return rng.normal(0.0, scale, genome_size(layers))
