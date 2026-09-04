import numpy as np

import config as cfg


def genome_size(n_in=None, hidden=None, n_out=None):
    n_in = cfg.N_IN if n_in is None else n_in
    hidden = cfg.HIDDEN if hidden is None else hidden
    n_out = cfg.N_OUT if n_out is None else n_out
    return n_in * hidden + hidden + hidden * n_out + n_out


def forward(pop, obs, n_in=None, hidden=None, n_out=None):
    n_in = cfg.N_IN if n_in is None else n_in
    hidden = cfg.HIDDEN if hidden is None else hidden
    n_out = cfg.N_OUT if n_out is None else n_out
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
