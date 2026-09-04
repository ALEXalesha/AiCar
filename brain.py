import numpy as np

import config as cfg


def genome_size(n_in=cfg.N_IN, hidden=cfg.HIDDEN, n_out=cfg.N_OUT):
    return n_in * hidden + hidden + hidden * n_out + n_out


def forward(pop, obs, n_in=cfg.N_IN, hidden=cfg.HIDDEN, n_out=cfg.N_OUT):
    need = genome_size(n_in, hidden, n_out)
    if pop.shape[1] != need:
        raise ValueError(f"геном длины {pop.shape[1]}, для сети {n_in}-{hidden}-{n_out} нужно {need}")

    k = n_in * hidden
    w1 = pop[:, :k].reshape(-1, n_in, hidden)
    b1 = pop[:, k:k + hidden]
    k += hidden
    w2 = pop[:, k:k + hidden * n_out].reshape(-1, hidden, n_out)
    k += hidden * n_out
    b2 = pop[:, k:k + n_out]

    h = np.tanh(np.einsum("ni,nih->nh", obs, w1) + b1)
    return np.tanh(np.einsum("nh,nho->no", h, w2) + b2)
