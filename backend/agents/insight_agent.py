def run(profile, cleaning, eda, statistics, ml):
    insights = []

    # 1. Missingness & Duplicates
    missing_total = profile.get("missing_total", 0)
    if missing_total > 0:
        col_count = len(profile.get("missing_by_column", {}))
        insights.append(f"Data Quality: Identified {missing_total:,} missing values across {col_count} columns.")
    else:
        insights.append("Data Quality: The dataset is complete with 0 missing values detected.")

    duplicates = profile.get("duplicates", 0)
    if duplicates > 0:
        insights.append(f"Deduplication: Found {duplicates:,} exact duplicate records requiring removal.")
    else:
        insights.append("Deduplication: Dataset contains unique records with 0 duplicate rows.")

    # 2. Key EDA Findings
    for finding in eda.get("findings", [])[:3]:
        insights.append(f"Pattern: {finding}")

    # 3. Correlations
    corrs = sorted(statistics.get("correlations", []), key=lambda x: abs(x["r"]), reverse=True)
    if corrs:
        top_corr = corrs[0]
        insights.append(
            f"Correlation: Strongest relationship measured between '{top_corr['a']}' and '{top_corr['b']}' "
            f"(r = {top_corr['r']:.2f}, p-value = {top_corr['p_value']:.4f})."
        )

    # 4. ML Takeaways & Leaderboard Winner
    if ml.get("best_model"):
        insights.append(
            f"Machine Learning Leaderboard: Top model is '{ml.get('best_model')}' "
            f"(score: {ml.get('best_score')}) across {ml.get('model_count', len(ml.get('models', [])))} evaluated algorithms."
        )
    elif ml.get("task") == "clustering":
        insights.append(f"Segmentation: Identified {ml.get('clusters')} natural clusters (Silhouette Score = {ml.get('silhouette')}).")

    return insights
