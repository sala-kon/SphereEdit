import torch, math
from collections import defaultdict

def _entropy_peaky(mask: torch.Tensor) -> float:
    # mask: (B=1,1,H,W) in [0,1]
    p = mask.flatten().float()
    s = p.sum().clamp_min(1e-6)
    p = (p / s).clamp_min(1e-8)                 # normalized
    H = -(p * (p + 1e-8).log()).sum().item()    # entropy
    H_max = math.log(p.numel() + 1e-8)
    return float(1.0 - H / H_max)               # peaky score in [0,1]

def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.view(1,-1), b.view(1,-1)).item())

def _iou_bin(a: torch.Tensor, b: torch.Tensor, tau: float) -> float:
    A = (a > tau).float(); B = (b > tau).float()
    inter = (A * B).sum().item()
    union = (A + B).clamp_max(1.0).sum().item() + 1e-6
    return float(inter / union)

class AdaptiveWindowEstimator:
    """
    Online hysteresis + (optional) priors to decide when to start/stop each attribute.
    Use .update(i, t, d_eps, mask) inside your timestep loop, then query .windows().
    """
    def __init__(self, n_attr, T,
                 start_thr=0.55, stop_thr=0.45, stable_k=2,
                 min_width=6, max_width=22,  # safety rails
                 priors=None,                # dict{name: (tmin,tmax)} or None
                 names=None,                 # list[str] len=n_attr
                 iou_tau=0.75, ema=0.6,
                 alpha=0.6, beta=0.4, gamma=0.2):
        self.n_attr, self.T = n_attr, T
        self.start_thr, self.stop_thr, self.stable_k = start_thr, stop_thr, stable_k
        self.min_width, self.max_width = min_width, max_width
        self.priors = priors or {}
        self.names = names or [f"a{k}" for k in range(n_attr)]
        self.iou_tau = iou_tau
        self.ema = ema
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        # per-attr state
        self.prev = [None]*n_attr
        self.running_min = [float("inf")]*n_attr
        self.running_max = [float("-inf")]*n_attr
        self.conf = [[] for _ in range(n_attr)]
        self.conf_ema = [0.0]*n_attr
        self.above_count = [0]*n_attr
        self.started = [False]*n_attr
        self.ended = [False]*n_attr
        self.start_idx = [None]*n_attr
        self.end_idx = [None]*n_attr

    def _norm01(self, a_idx, x):
        self.running_min[a_idx] = min(self.running_min[a_idx], x)
        self.running_max[a_idx] = max(self.running_max[a_idx], x)
        lo, hi = self.running_min[a_idx], self.running_max[a_idx]
        if hi <= lo + 1e-8: return 0.0
        return float((x - lo) / (hi - lo))

    def update(self, a_idx, t, d_eps: torch.Tensor, mask: torch.Tensor, name=None):
        """
        Returns: bool -> should attribute a_idx be ACTIVE at step t (online hysteresis).
        Call this once per attribute when you have direction & mask.
        """
        # strength
        d = float(d_eps.norm().item() / d_eps.numel())
        d01 = self._norm01(a_idx, d)

        # sharpness
        s = _entropy_peaky(mask)   # higher = peakier

        # stability
        if self.prev[a_idx] is None:
            cosv, iou = 1.0, 1.0
        else:
            cosv = _cos_sim(d_eps.detach(), self.prev[a_idx]["d_eps"])
            iou  = _iou_bin(mask.detach(), self.prev[a_idx]["mask"], self.iou_tau)

        # composite confidence with EMA
        conf_raw = self.alpha*d01 + self.beta*s + self.gamma*0.5*(cosv + iou)
        self.conf_ema[a_idx] = self.ema*self.conf_ema[a_idx] + (1-self.ema)*conf_raw
        self.conf[a_idx].append(self.conf_ema[a_idx])

        # store prev
        self.prev[a_idx] = {"d_eps": d_eps.detach(), "mask": mask.detach()}

        # prior clamp (optional)
        nm = (name or self.names[a_idx])
        tmin, tmax = self.priors.get(nm, (0, self.T-1))

        # hysteresis logic
        active = False
        if not self.started[a_idx]:
            if t < tmin or self.ended[a_idx]:
                return False
            if self.conf_ema[a_idx] >= self.start_thr:
                self.above_count[a_idx] += 1
            else:
                self.above_count[a_idx] = 0
            if self.above_count[a_idx] >= self.stable_k:
                self.started[a_idx] = True
                self.start_idx[a_idx] = max(t - self.stable_k + 1, tmin)
                active = True
        else:
            # already started
            if (self.conf_ema[a_idx] < self.stop_thr and
                (t - self.start_idx[a_idx] + 1) >= self.min_width) or t >= tmax:
                self.ended[a_idx] = True
                self.end_idx[a_idx] = max(self.start_idx[a_idx],
                                          min(t-1, self.start_idx[a_idx] + self.max_width - 1, tmax))
                active = False
            else:
                active = True
        return active

    def windows(self):
        # finalize unfinished windows
        for k in range(self.n_attr):
            if self.started[k] and not self.ended[k]:
                self.end_idx[k] = min(self.T-1, self.start_idx[k] + self.max_width - 1)
                self.ended[k] = True
        return [(self.start_idx[k], self.end_idx[k]) for k in range(self.n_attr)]
