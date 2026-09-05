import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

import config as cfg
import dataset
import track

MODEL_FILE = cfg.MODEL_FILE
LATENT = 12
HIDDEN = (256, 128)


class VAE(nn.Module):
    def __init__(self, n_points=cfg.TRACK_POINTS, latent=LATENT, hidden=HIDDEN):
        super().__init__()
        h1, h2 = hidden
        self.encode = nn.Sequential(nn.Linear(n_points, h1), nn.ReLU(),
                                    nn.Linear(h1, h2), nn.ReLU())
        self.to_mu = nn.Linear(h2, latent)
        self.to_logvar = nn.Linear(h2, latent)
        self.decode = nn.Sequential(nn.Linear(latent, h2), nn.ReLU(),
                                    nn.Linear(h2, h1), nn.ReLU(),
                                    nn.Linear(h1, n_points), nn.Sigmoid())

    def forward(self, x):
        h = self.encode(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


def augment(batch, rng):
    shifts = rng.integers(0, batch.shape[1], len(batch))
    rolled = np.stack([np.roll(row, s) for row, s in zip(batch, shifts)])
    flip = rng.random(len(batch)) < 0.5
    rolled[flip] = rolled[flip, ::-1]
    return rolled


def train(profiles, epochs=400, batch_size=128, lr=1e-3, beta=1.0, seed=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = VAE(profiles.shape[1]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        order = rng.permutation(len(profiles))
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = augment(profiles[order[start:start + batch_size]], rng)
            x = torch.tensor(batch, dtype=torch.float32, device=device)

            out, mu, logvar = model(x)
            recon = ((out - x) ** 2).sum(dim=1).mean()
            kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=1)).mean()
            loss = recon + beta * kl

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss) * len(batch)
            seen += len(batch)

        history.append(total / seen)
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  эпоха {epoch + 1:4d}  ошибка {history[-1]:8.4f}")
    return model, history


def export(model, path=MODEL_FILE):
    weights = {}
    for i, layer in enumerate(model.decode):
        if isinstance(layer, nn.Linear):
            weights[f"w{i}"] = layer.weight.detach().cpu().numpy().T
            weights[f"b{i}"] = layer.bias.detach().cpu().numpy()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, latent=np.array([LATENT]), n_points=np.array([cfg.TRACK_POINTS]), **weights)
    return path


def evaluate(sample_fn, n=200, width=None):
    width = cfg.TRACK_WIDTH if width is None else width
    good, scores = 0, []
    for radii in sample_fn(n):
        center = track.centerline_from_radii(radii)
        if track.walls_are_sane(center, width) and cfg.LEN_MIN <= track.polyline_length(center) <= cfg.LEN_MAX:
            good += 1
            scores.append(track.interest(center))
    return good / n, np.array(scores)


def main():
    ap = argparse.ArgumentParser(description="Обучение генератора трасс")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=dataset.DATASET_FILE)
    ap.add_argument("--out", default=MODEL_FILE)
    args = ap.parse_args()

    profiles, scores = dataset.load(args.data)
    print(f"датасет: {len(profiles)} профилей, интерес {scores.min():.1f}..{scores.max():.1f}, "
          f"медиана {np.median(scores):.1f}")

    t0 = time.perf_counter()
    model, history = train(profiles, epochs=args.epochs, beta=args.beta, seed=args.seed)
    print(f"обучено за {time.perf_counter() - t0:.0f} с, ошибка {history[0]:.2f} -> {history[-1]:.2f}")

    export(model, args.out)
    print(f"веса декодера записаны в {args.out}")

    import trackgen
    gen = trackgen.Generator(args.out)
    rng = np.random.default_rng(1)
    rate, sampled = evaluate(lambda n: [gen.sample_radii(rng) for _ in range(n)])
    print()
    print(f"проверка: годных {rate:.0%}")
    if len(sampled):
        print(f"интерес сгенерированных: медиана {np.median(sampled):.1f}, "
              f"{np.percentile(sampled, 25):.1f}..{np.percentile(sampled, 75):.1f}, "
              f"макс {sampled.max():.1f}")


if __name__ == "__main__":
    main()
