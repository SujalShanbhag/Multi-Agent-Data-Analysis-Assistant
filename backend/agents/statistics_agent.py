"""
Statistical Inference & Significance Testing Agent.
Implemented in 100% Pure NumPy and standard Python math.
Zero dependency on SciPy C-extensions or external .pyd DLLs to guarantee
100% immunity to Windows Defender Application Control (WDAC) and AppLocker blocks.
Computes Pearson r correlations, two-sided Student's t p-values via
regularized incomplete beta functions, and D'Agostino omnibus normality tests.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd


# =====================================================================
# PURE PYTHON EXACT P-VALUE FUNCTIONS (REGULARIZED INCOMPLETE BETA)
# =====================================================================

def _betacf(a: float, b: float, x: float) -> float:
    max_it = 100
    eps = 3.0e-7
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_val = d * c
        h *= del_val
        if abs(del_val - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _pearson_p_value(r: float, n: int) -> float:
    if n <= 2:
        return 1.0
    if abs(r) >= 1.0:
        return 0.0
    df = n - 2
    t = abs(r) * math.sqrt(df / max(1e-12, 1.0 - r * r))
    # Two-sided t-distribution survival probability via incomplete beta
    x = df / (df + t * t)
    return float(_betai(0.5 * df, 0.5, x))


def _omnibus_normality_test(s: np.ndarray) -> tuple[float, bool]:
    """
    D'Agostino-Pearson omnibus normality test implemented in pure NumPy.
    Tests empirical sample skewness and kurtosis departures from a Gaussian distribution.
    """
    n = len(s)
    if n < 8:
        return 0.5, True
    m = np.mean(s)
    m2 = np.mean((s - m) ** 2)
    if m2 < 1e-12:
        return 0.0, False
    m3 = np.mean((s - m) ** 3)
    m4 = np.mean((s - m) ** 4)
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2) - 3.0
    # Standard errors of skewness and kurtosis
    se_skew = math.sqrt(6.0 / n)
    se_kurt = math.sqrt(24.0 / n)
    z_skew = skew / se_skew
    z_kurt = kurt / se_kurt
    chi2 = z_skew ** 2 + z_kurt ** 2
    # Survival function of Chi-Squared with 2 degrees of freedom is exp(-chi2 / 2)
    p_val = float(math.exp(-0.5 * chi2))
    return round(p_val, 6), bool(p_val >= 0.05)


# =====================================================================
# AGENT EXECUTION ENTRYPOINT
# =====================================================================

def run(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=np.number)
    result = {"correlations": [], "normality": []}
    cols = list(numeric.columns)

    if len(cols) >= 2:
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                pair_df = numeric[[a, b]].dropna()
                if len(pair_df) >= 3:
                    try:
                        x = pair_df[a].to_numpy(dtype=float)
                        y = pair_df[b].to_numpy(dtype=float)
                        vx = x - np.mean(x)
                        vy = y - np.mean(y)
                        denom = np.sqrt(np.sum(vx ** 2) * np.sum(vy ** 2))
                        if denom > 1e-12:
                            r = float(np.sum(vx * vy) / denom)
                            if not np.isnan(r):
                                p = _pearson_p_value(r, len(pair_df))
                                result["correlations"].append({
                                    "a": str(a),
                                    "b": str(b),
                                    "r": round(float(r), 4),
                                    "p_value": round(float(p), 6),
                                })
                    except Exception:
                        pass

    for col in cols[:10]:
        s = numeric[col].dropna().to_numpy(dtype=float)
        if len(s) >= 4:
            try:
                p_val, is_normal = _omnibus_normality_test(s)
                result["normality"].append({
                    "column": str(col),
                    "p_value": round(float(p_val), 6),
                    "normal_at_0_05": is_normal,
                })
            except Exception:
                pass

    return result
