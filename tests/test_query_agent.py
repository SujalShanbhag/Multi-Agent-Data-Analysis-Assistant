import pandas as pd
from backend.agents.query_agent import QueryPlan, execute_plan, answer_question

def test_dynamic_query_execution():
    df = pd.DataFrame({
        "Fruit": ["Apple", "Apple", "Banana", "Banana"],
        "Sales": [10, 20, 5, 30],
    })

    plan = QueryPlan(
        operation="group_aggregate",
        column="Sales",
        group_by="Fruit",
        aggregation="sum",
        n=5,
        sort_desc=True,
    )

    result = execute_plan(df, plan)
    assert "Banana" in result["answer"]
    assert "35" in result["answer"]

def test_correlation_execution():
    df = pd.DataFrame({
        "Price": [1, 2, 3, 4],
        "Sales": [2, 4, 6, 8],
    })

    plan = QueryPlan(
        operation="correlation",
        column="Price",
        second_column="Sales",
    )

    result = execute_plan(df, plan)
    assert "1" in result["answer"]

def test_local_query_fallback():
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet"],
        "Revenue": [1000, 500, 300],
    })
    # Offline / local answer
    res = answer_question(df, "What is the total revenue?")
    assert "Revenue" in res["answer"]
    assert "1,800" in res["answer"] or "1800" in res["answer"]
