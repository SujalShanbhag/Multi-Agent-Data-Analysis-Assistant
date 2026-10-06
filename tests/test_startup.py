def test_backend_import():
    import backend.main
    assert backend.main.app is not None

def test_workflow():
    from backend.workflows.analysis_graph import run_workflow
    import pandas as pd
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "B": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "Target": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    })
    res = run_workflow(df, "test", "Target")
    assert "profile" in res
    assert "cleaning" in res
    assert "eda" in res
    assert "statistics" in res
    assert "ml" in res
    assert "validation" in res
    assert "insights" in res
    assert "report" in res
