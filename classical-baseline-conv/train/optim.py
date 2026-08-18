"""
train/optim.py — a minimal Adam optimizer, dependency-free (plain numpy).

Identical to classical-baseline-mlp/train/optim.py, duplicated here (not
imported cross-directory) so this directory stays self-contained apart from
its one deliberate dependency on the shared QML data pipeline (see
trainer.py's docstring). Standard Adam, matching PennyLane's AdamOptimizer
defaults (betas 0.9/0.999, eps=1e-8) for behavioural comparability with the
QML side's own optimizer without importing PennyLane itself.
"""
import numpy as np


class Adam:
    def __init__(self, params, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)
        self.t = 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad ** 2)
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    x = rng.randn(5)
    opt = Adam(x, lr=0.1)
    for _ in range(500):
        g = 2 * (x - 3.0)
        x = opt.step(x, g)
    err = np.max(np.abs(x - 3.0))
    print(f"Adam smoke test -> converged to x={np.round(x,4)}  max|err|={err:.2e}  "
          f"{'OK' if err < 1e-3 else 'FAIL'}")
