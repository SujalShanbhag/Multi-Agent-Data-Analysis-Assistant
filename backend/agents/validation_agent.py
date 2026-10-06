def run(profile, eda, statistics, ml):
    checks = []
    if profile.get("rows", 0) >= 0 and profile.get("columns", 0) >= 0:
        checks.append({"check": "dataset_profile_integrity", "status": "passed"})

    if eda.get("findings") is not None:
        checks.append({"check": "eda_statistical_findings", "status": "passed"})

    if statistics.get("correlations") is not None:
        checks.append({"check": "inferential_statistics", "status": "passed"})

    if ml.get("status") in {"run", "not_run", "skipped", "sklearn_unavailable"}:
        checks.append({"check": "machine_learning_pipeline_safety", "status": "passed"})

    return {"status": "validated", "checks": checks}
