"""Process capability statistics.

Short-term (Cp/Cpk) vs long-term (Pp/Ppk)
-----------------------------------------
* sigma_within  - short-term, *within-subgroup* variation, estimated from the
                  moving range (individuals, subgroup size 1) or the average
                  subgroup range (subgroup size 2-10).  It ignores drift
                  between subgroups, so it describes the process's potential.
* sigma_overall - long-term, ordinary sample standard deviation of ALL data.
                  It includes drift, shifts and between-subgroup variation.

Cp/Cpk use sigma_within; Pp/Ppk use sigma_overall.  If the process is stable
and centred, the two families agree; a Cpk >> Ppk gap signals instability.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# ── Control-chart constants (indexed by subgroup size n) ─────────────────────
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
      7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483,
      7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
D3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0,
      7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004,
      7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}

MIN_RECOMMENDED_N = 30


# ── Result containers ────────────────────────────────────────────────────────
@dataclass
class Index:
    value: float
    lo: float
    hi: float


@dataclass
class ControlChart:
    kind: str                      # "I-MR" or "Xbar-R"
    x: np.ndarray                  # plotted points (individuals or subgroup means)
    x_cl: float
    x_ucl: float
    x_lcl: float
    disp: np.ndarray               # moving ranges or subgroup ranges
    disp_cl: float
    disp_ucl: float
    disp_lcl: float
    violations: dict = field(default_factory=dict)   # rule name -> list of point indices (top chart)
    disp_alarms: list = field(default_factory=list)  # indices of range / moving-range points beyond limits

    # Only these decide "stable". The run/trend rules have a noticeable false-alarm
    # rate on long series, so they are reported as signals to investigate instead.
    ALARM_RULES = ("1 point beyond 3σ",)

    @property
    def alarms(self):
        pts = set()
        for k in self.ALARM_RULES:
            pts.update(self.violations.get(k, []))
        return sorted(pts)

    @property
    def signals(self):
        pts = set()
        for k, v in self.violations.items():
            if k not in self.ALARM_RULES:
                pts.update(v)
        return sorted(pts)

    @property
    def n_alarms(self):
        return len(self.alarms) + len(self.disp_alarms)

    @property
    def stable(self):
        return self.n_alarms == 0


# ── Helpers ──────────────────────────────────────────────────────────────────
def make_subgroups(data, n):
    """Consecutive rows -> subgroups of size n (incomplete trailing group dropped)."""
    data = np.asarray(data, dtype=float)
    k = len(data) // n
    return data[: k * n].reshape(k, n)


def sigma_within(data, subgroup_size=1):
    """Short-term sigma: MR-bar/d2 (n=1) or R-bar/d2 (n>=2)."""
    data = np.asarray(data, dtype=float)
    if subgroup_size == 1:
        mr = np.abs(np.diff(data))
        return float(mr.mean() / D2[2]) if len(mr) else np.nan
    groups = make_subgroups(data, subgroup_size)
    if len(groups) < 2:
        return np.nan
    rbar = np.ptp(groups, axis=1).mean()
    return float(rbar / D2[subgroup_size])


def _safe_div(a, b):
    return a / b if b and np.isfinite(b) and b > 0 else np.nan


def _capability(mean, sigma, usl, lsl):
    """Return (Cp, Cpl, Cpu, Cpk) for a given sigma."""
    cp = _safe_div(usl - lsl, 6 * sigma)
    cpu = _safe_div(usl - mean, 3 * sigma)
    cpl = _safe_div(mean - lsl, 3 * sigma)
    return cp, cpl, cpu, np.nanmin([cpl, cpu]) if not (np.isnan(cpl) and np.isnan(cpu)) else np.nan


def cp_ci(cp, n, conf=0.95):
    """Chi-square confidence interval for Cp / Pp (assumes normal data)."""
    a = 1 - conf
    df = n - 1
    lo = cp * np.sqrt(stats.chi2.ppf(a / 2, df) / df)
    hi = cp * np.sqrt(stats.chi2.ppf(1 - a / 2, df) / df)
    return lo, hi


def cpk_ci(cpk, n, conf=0.95):
    """Bissell's approximation for the CI of Cpk / Ppk (assumes normal data)."""
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    half = z * np.sqrt(1 / (9 * n) + cpk ** 2 / (2 * (n - 1)))
    return cpk - half, cpk + half


# ── Normality ────────────────────────────────────────────────────────────────
def anderson_darling(data):
    """Anderson-Darling statistic A^2 and p-value (D'Agostino & Stephens approximation)."""
    n = len(data)
    x = np.sort(np.asarray(data, dtype=float))
    sd = x.std(ddof=1)
    if sd == 0:
        return np.inf, 0.0
    cdf = np.clip(stats.norm.cdf((x - x.mean()) / sd), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    a2 = float(-n - np.mean((2 * i - 1) * (np.log(cdf) + np.log(1 - cdf[::-1]))))
    a2s = a2 * (1 + 0.75 / n + 2.25 / n ** 2)
    if a2s >= 0.6:
        p = np.exp(1.2937 - 5.709 * a2s + 0.0186 * a2s ** 2)
    elif a2s >= 0.34:
        p = np.exp(0.9177 - 4.279 * a2s - 1.38 * a2s ** 2)
    elif a2s >= 0.2:
        p = 1 - np.exp(-8.318 + 42.796 * a2s - 59.938 * a2s ** 2)
    else:
        p = 1 - np.exp(-13.436 + 101.14 * a2s - 223.73 * a2s ** 2)
    return a2, float(np.clip(p, 0.0, 1.0))


def normality_tests(data):
    data = np.asarray(data, dtype=float)
    a2, p_ad = anderson_darling(data)
    out = {"ad_stat": a2, "ad_p": p_ad, "sw_stat": np.nan, "sw_p": np.nan}
    if 3 <= len(data) <= 5000:
        w, p = stats.shapiro(data)
        out.update(sw_stat=float(w), sw_p=float(p))
    out["is_normal"] = p_ad >= 0.05
    out["skew"] = float(stats.skew(data))
    out["excess_kurtosis"] = float(stats.kurtosis(data))
    return out


def qq_points(data):
    """Theoretical vs sample quantiles + fitted reference line."""
    data = np.sort(np.asarray(data, dtype=float))
    n = len(data)
    q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    slope, intercept = np.std(data, ddof=1), np.mean(data)
    return q, data, slope, intercept


# ── Control chart ────────────────────────────────────────────────────────────
def _rule_violations(x, cl, sigma_x):
    """Western Electric / Nelson tests 1-3 on the plotted points."""
    x = np.asarray(x)
    ucl, lcl = cl + 3 * sigma_x, cl - 3 * sigma_x
    v = {}
    v["1 point beyond 3σ"] = [int(i) for i in np.where((x > ucl) | (x < lcl))[0]]

    side = np.sign(x - cl)
    run8 = set()
    for i in range(len(x) - 7):
        w = side[i:i + 8]
        if np.all(w > 0) or np.all(w < 0):
            run8.update(range(i, i + 8))
    v["8 in a row on one side"] = sorted(int(i) for i in run8)

    d = np.sign(np.diff(x))
    trend = set()
    for i in range(len(d) - 4):
        w = d[i:i + 5]
        if np.all(w > 0) or np.all(w < 0):
            trend.update(range(i, i + 6))
    v["6 in a row trending"] = sorted(int(i) for i in trend)
    return v


def control_chart(data, subgroup_size=1):
    data = np.asarray(data, dtype=float)
    if subgroup_size == 1:
        mr = np.abs(np.diff(data))
        mrbar = mr.mean()
        cl = data.mean()
        sig_x = mrbar / D2[2]
        v = _rule_violations(data, cl, sig_x)
        ch = ControlChart("I-MR", data, cl, cl + 3 * sig_x, cl - 3 * sig_x,
                          mr, mrbar, D4[2] * mrbar, 0.0, v)
        ch.disp_alarms = [int(i) for i in np.where(mr > ch.disp_ucl)[0]]
        return ch

    g = make_subgroups(data, subgroup_size)
    means = g.mean(axis=1)
    ranges = np.ptp(g, axis=1)
    xbarbar, rbar = means.mean(), ranges.mean()
    n = subgroup_size
    ucl, lcl = xbarbar + A2[n] * rbar, xbarbar - A2[n] * rbar
    v = _rule_violations(means, xbarbar, (ucl - xbarbar) / 3)
    ch = ControlChart("Xbar-R", means, xbarbar, ucl, lcl,
                      ranges, rbar, D4[n] * rbar, D3[n] * rbar, v)
    ch.disp_alarms = [int(i) for i in np.where((ranges > ch.disp_ucl) | (ranges < ch.disp_lcl))[0]]
    return ch


# ── Main entry point ─────────────────────────────────────────────────────────
def calculate_metrics(data, usl, lsl, subgroup_size=1, conf=0.95):
    """Full capability analysis. Returns a dict (see keys below)."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)
    mean = float(np.mean(data))
    s_within = sigma_within(data, subgroup_size)
    s_overall = float(np.std(data, ddof=1))

    cp, cpl, cpu, cpk = _capability(mean, s_within, usl, lsl)
    pp, ppl, ppu, ppk = _capability(mean, s_overall, usl, lsl)

    def mk(value, kind):
        if not np.isfinite(value):
            return Index(np.nan, np.nan, np.nan)
        lo, hi = (cp_ci if kind == "cp" else cpk_ci)(value, n, conf)
        return Index(float(value), float(lo), float(hi))

    # Expected / observed nonconformance (ppm)
    def ppm(sigma):
        if not np.isfinite(sigma) or sigma <= 0:
            return np.nan, np.nan
        below = stats.norm.cdf(lsl, mean, sigma) * 1e6
        above = stats.norm.sf(usl, mean, sigma) * 1e6
        return below, above

    w_lo, w_hi = ppm(s_within)
    o_lo, o_hi = ppm(s_overall)
    obs_lo = float(np.sum(data < lsl) / n * 1e6)
    obs_hi = float(np.sum(data > usl) / n * 1e6)

    total_overall = o_lo + o_hi
    z_bench = (float(stats.norm.isf(total_overall / 1e6))
               if np.isfinite(total_overall) and 0 < total_overall < 1e6 else np.nan)

    norm = normality_tests(data)

    # Non-parametric (percentile / ISO 22514) Ppk for non-normal data
    p_lo, med, p_hi = np.percentile(data, [0.135, 50, 99.865])
    pp_np = _safe_div(usl - lsl, p_hi - p_lo)
    ppk_np = np.nanmin([_safe_div(usl - med, p_hi - med), _safe_div(med - lsl, med - p_lo)])

    chart = control_chart(data, subgroup_size)

    return {
        "n": n, "subgroup_size": subgroup_size, "conf": conf,
        "mean": mean, "median": float(med),
        "sigma_within": s_within, "sigma_overall": s_overall,
        "Cp": mk(cp, "cp"), "Cpk": mk(cpk, "cpk"), "Cpl": cpl, "Cpu": cpu,
        "Pp": mk(pp, "cp"), "Ppk": mk(ppk, "cpk"), "Ppl": ppl, "Ppu": ppu,
        "Pp_np": float(pp_np), "Ppk_np": float(ppk_np),
        "ppm_within": (w_lo, w_hi), "ppm_overall": (o_lo, o_hi),
        "ppm_observed": (obs_lo, obs_hi),
        "z_bench": z_bench,
        "n_out_of_spec": int(np.sum((data < lsl) | (data > usl))),
        "normality": norm,
        "chart": chart,
        "stable": chart.stable,
    }


def get_verdict(idx, target=1.33):
    """Verdict that accounts for sampling uncertainty, not just the point estimate.

    idx: Index for the headline metric (Cpk).  Returns (status, icon, message).
    """
    if not np.isfinite(idx.value):
        return "UNKNOWN", "❔", "Not enough variation or data to compute capability."
    if idx.lo >= target:
        return "PASS", "✅", (f"Capable — Cpk {idx.value:.2f}, and even the lower confidence "
                              f"bound ({idx.lo:.2f}) clears the target of {target:g}.")
    if idx.value >= target:
        return "WARNING", "⚠️", (f"Meets target on the point estimate (Cpk {idx.value:.2f}), but the "
                                 f"lower confidence bound ({idx.lo:.2f}) falls below {target:g}. "
                                 f"More data is needed to demonstrate capability.")
    if idx.value >= 1.0:
        return "WARNING", "⚠️", f"Marginal — Cpk {idx.value:.2f} is below the target of {target:g}."
    return "FAIL", "❌", f"Not capable — Cpk {idx.value:.2f} is below 1.00."
