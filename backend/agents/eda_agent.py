"""
Exploratory Data Analysis (EDA) Agent.
Performs distribution analysis, correlation mining, IQR outlier detection,
box plot quartile computation, 2D scatter plots, and time-series trend extraction.
Prepares structured chart payloads for interactive multi-dimensional visual analytics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _calculate_distributions(df: pd.DataFrame) -> dict:
    distributions = {}
    num_df = df.select_dtypes(include=np.number)

    for col in num_df.columns[:8]:
        s = num_df[col].dropna()
        if len(s) < 3:
            continue
        counts, bin_edges = np.histogram(s, bins=min(6, len(s)))
        bin_labels = [
            f"{bin_edges[i]:,.0f} - {bin_edges[i+1]:,.0f}" if abs(bin_edges[i+1]) >= 1000
            else f"{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}"
            for i in range(len(counts))
        ]
        distributions[str(col)] = {
            "labels": bin_labels,
            "counts": [int(c) for c in counts],
            "mean": float(np.mean(s)),
            "median": float(np.median(s)),
            "min": float(np.min(s)),
            "max": float(np.max(s)),
        }
    return distributions


def _calculate_category_breakdowns(df: pd.DataFrame) -> dict:
    breakdowns = {}
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    num_cols = df.select_dtypes(include=np.number).columns

    target_num = num_cols[-1] if len(num_cols) else None
    for priority in ["revenue", "sales", "price", "units"]:
        for c in num_cols:
            if priority in c.lower():
                target_num = c
                break
        if target_num and target_num != num_cols[-1]:
            break

    for col in cat_cols[:6]:
        vc = df[col].value_counts().head(7)
        entry = {
            "labels": [str(k) for k in vc.index],
            "counts": [int(v) for v in vc.values],
        }
        if target_num:
            avg_by_cat = df.groupby(col)[target_num].mean().reindex(vc.index).fillna(0.0)
            entry["metric_name"] = f"Avg {target_num}"
            entry["metric_values"] = [round(float(v), 2) for v in avg_by_cat.values]

        breakdowns[str(col)] = entry
    return breakdowns


def _calculate_boxplots(df: pd.DataFrame) -> dict:
    boxplots = {}
    num_df = df.select_dtypes(include=np.number)
    for col in num_df.columns[:8]:
        s = num_df[col].dropna()
        if len(s) < 4:
            continue
        q1, median, q3 = np.percentile(s, [25, 50, 75])
        iqr = q3 - q1
        min_w = float(np.min(s[s >= (q1 - 1.5 * iqr)])) if len(s[s >= (q1 - 1.5 * iqr)]) else float(np.min(s))
        max_w = float(np.max(s[s <= (q3 + 1.5 * iqr)])) if len(s[s <= (q3 + 1.5 * iqr)]) else float(np.max(s))
        boxplots[str(col)] = {
            "min": round(min_w, 2),
            "q1": round(float(q1), 2),
            "median": round(float(median), 2),
            "q3": round(float(q3), 2),
            "max": round(max_w, 2),
            "mean": round(float(np.mean(s)), 2),
        }
    return boxplots


def _calculate_scatter_pairs(df: pd.DataFrame) -> dict:
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return {}
    cols = list(num_df.columns[:6])
    scatter_data = {}
    for i in range(len(cols) - 1):
        x_col = cols[i]
        y_col = cols[i + 1]
        pair_df = num_df[[x_col, y_col]].dropna().head(100)
        scatter_data[f"{x_col} vs {y_col}"] = {
            "x_label": x_col,
            "y_label": y_col,
            "points": [{"x": round(float(row[x_col]), 2), "y": round(float(row[y_col]), 2)} for _, row in pair_df.iterrows()]
        }
    return scatter_data


def _calculate_time_series(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include=np.number).columns
    if not len(num_cols):
        return {}
    val_col = num_cols[0]
    for prio in ["revenue", "sales", "price", "profit", "units"]:
        for c in num_cols:
            if prio in c.lower():
                val_col = c
                break

    # Look for datetime columns
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "day" in c.lower() or "year" in c.lower()]
    if date_cols:
        d_col = date_cols[0]
        try:
            sub = df[[d_col, val_col]].dropna().copy()
            sub[d_col] = pd.to_datetime(sub[d_col], errors="coerce")
            sub = sub.dropna().sort_values(d_col).tail(30)
            if len(sub) >= 3:
                return {
                    "labels": [d.strftime("%Y-%m-%d") for d in sub[d_col]],
                    "values": [round(float(v), 2) for v in sub[val_col]],
                    "val_col": str(val_col),
                    "date_col": str(d_col),
                }
        except Exception:
            pass

    # Fallback to sequential row trends
    sample_sub = df[val_col].dropna().head(25)
    return {
        "labels": [f"Row {i+1}" for i in range(len(sample_sub))],
        "values": [round(float(v), 2) for v in sample_sub],
        "val_col": str(val_col),
        "date_col": "Index Sequence",
    }


def run(df: pd.DataFrame) -> dict:
    findings = []
    num_df = df.select_dtypes(include=np.number)
    correlations_chart = []

    # 1. Correlations
    if num_df.shape[1] >= 2:
        cols = list(num_df.columns)
        pairs = []
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                valid = num_df[[a, b]].dropna()
                if len(valid) >= 3:
                    r = float(valid[a].corr(valid[b]))
                    if not np.isnan(r):
                        pairs.append((abs(r), r, str(a), str(b)))

        pairs.sort(reverse=True)
        for _, r_val, a, b in pairs[:6]:
            strength = "strong" if abs(r_val) >= 0.7 else "moderate" if abs(r_val) >= 0.4 else "weak"
            direction = "positive" if r_val >= 0 else "negative"
            findings.append(f"{a} and {b} show a {strength} {direction} correlation (r={r_val:.2f}).")
            correlations_chart.append({
                "pair": f"{a} vs {b}",
                "r": round(r_val, 3),
            })

    # 2. Skewness
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) > 2:
            skew = float(s.skew())
            if abs(skew) > 1.0:
                direction = "right-skewed" if skew > 0 else "left-skewed"
                findings.append(f"Column '{col}' is {direction} (skewness = {skew:.2f}).")

    # 3. Chart payloads for comprehensive visual analytics
    distributions = _calculate_distributions(df)
    category_charts = _calculate_category_breakdowns(df)
    boxplots = _calculate_boxplots(df)
    scatter_pairs = _calculate_scatter_pairs(df)
    time_series = _calculate_time_series(df)

    return {
        "findings": findings,
        "numeric_columns": [str(c) for c in num_df.columns],
        "visualizations": {
            "distributions": distributions,
            "correlations": correlations_chart,
            "categories": category_charts,
            "boxplots": boxplots,
            "scatter": scatter_pairs,
            "timeseries": time_series,
        },
    }


def outliers(df: pd.DataFrame) -> list[dict]:
    result = []
    num_df = df.select_dtypes(include=np.number)

    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            count = 0
            lower, upper = q1, q3
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((s < lower) | (s > upper)).sum())

        result.append({
            "column": str(col),
            "outliers": count,
            "lower_bound": round(float(lower), 2),
            "upper_bound": round(float(upper), 2),
        })
    return result
