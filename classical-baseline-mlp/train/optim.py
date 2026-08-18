"""
train/optim.py — a minimal Adam optimizer, dependency-free (plain numpy).

The QML side uses `qml.AdamOptimizer` (PennyLane's Adam, itself a standard
textbook implementation with default betas 0.9/0.999 and eps=1e-8). To keep
the classical baseline's optimizer *behaviourally* comparable (same update
rule, same defaults) without importing PennyLane into a directory that has
nothing else quantum in it, this reimplements standard Adam directly:
    m_t = b1*m_{t-1} + (1-b1)*g
    v_t = b2*v_{t-1} + (1-b2)*g^2
    m_hat = m_t / (1-b1^t) ; v_hat = v_t / (1-b2^t)
    theta -= lr * m_hat / (sqrt(v_hat) + eps)
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
    # smoke test: minimize a simple quadratic f(x) = sum((x-3)^2), grad = 2(x-3)
    rng = np.random.RandomState(0)
    x = rng.randn(5)
    opt = Adam(x, lr=0.1)
    for _ in range(500):
        g = 2 * (x - 3.0)
        x = opt.step(x, g)
    err = np.max(np.abs(x - 3.0))
    print(f"Adam smoke test -> converged to x={np.round(x,4)}  max|err|={err:.2e}  "
          f"{'OK' if err < 1e-3 else 'FAIL'}")
