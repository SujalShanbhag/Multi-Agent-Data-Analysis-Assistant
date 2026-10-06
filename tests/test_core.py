import pandas as pd
from backend.tools.data_tools import profile_dataframe, cleaned_dataframe, numeric_summary, categorical_summary

def test_profile():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "x", "y"]})
    p = profile_dataframe(df)
    assert p["rows"] == 3
    assert p["missing_total"] == 1
    assert p["columns"] == 2
    assert "a" in p["missing_by_column"]

def test_clean():
    df = pd.DataFrame({"a": [1, 1, None], "b": ["cat", "cat", "dog"]})
    out = cleaned_dataframe(df)
    assert len(out) == 2  # duplicate removed
    assert out["a"].isna().sum() == 0  # missing filled

def test_summaries():
    df = pd.DataFrame({"num": [10, 20, 30], "cat": ["alpha", "beta", "alpha"]})
    n_sum = numeric_summary(df)
    c_sum = categorical_summary(df)
    assert len(n_sum) == 1
    assert n_sum[0]["column"] == "num"
    assert len(c_sum) == 1
    assert c_sum[0]["column"] == "cat"
