"""
Natural Language Query Agent with Dynamic Gemini Multi-Model Discovery
and resilient Local Analytics Fallback.
Discovers available Gemini models dynamically from the Google GenAI SDK
without any hardcoding.
"""

from __future__ import annotations
import difflib
import json
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

Operation = Literal[
    "dataset_summary",
    "row_count",
    "column_count",
    "data_types",
    "missing",
    "duplicates",
    "unique",
    "frequency",
    "aggregate",
    "group_aggregate",
    "top_n",
    "bottom_n",
    "correlation",
    "filter_count",
    "filtered_rows",
]


class FilterCondition(BaseModel):
    column: str = Field(description="Exact dataset column name.")
    operator: Literal[
        "equals",
        "not_equals",
        "contains",
        "greater_than",
        "less_than",
        "greater_or_equal",
        "less_or_equal",
    ] = "equals"
    value: str = Field(description="Filter value represented as text.")


class QueryPlan(BaseModel):
    operation: Operation
    column: Optional[str] = Field(default=None, description="Exact dataset column name.")
    group_by: Optional[str] = Field(default=None, description="Exact grouping column name.")
    second_column: Optional[str] = Field(default=None, description="Second column for correlation.")
    aggregation: Optional[
        Literal["sum", "mean", "median", "min", "max", "std", "var", "count"]
    ] = None
    n: int = Field(default=5, ge=1, le=20)
    sort_desc: bool = True
    filters: list[FilterCondition] = Field(default_factory=list)
    explanation: str = Field(default="Query plan generated for dataset.")


# =====================================================================
# DYNAMIC GEMINI MODEL DISCOVERY (Zero Hardcoding)
# =====================================================================

def _get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def discover_gemini_models(client=None) -> list[str]:
    """
    Dynamically queries Google GenAI API to discover all available Gemini models
    capable of content generation. Zero hardcoded model names.
    """
    if client is None:
        client = _get_gemini_client()
    if client is None:
        return []

    discovered = []
    try:
        model_list = list(client.models.list())
        for m in model_list:
            name = getattr(m, "name", "") or ""
            clean_name = name.removeprefix("models/").strip()
            # Only select generative Gemini models, filtering out embeddings or non-text models
            if "gemini" in clean_name.lower():
                if not any(skip in clean_name.lower() for skip in ["embedding", "imagen", "veo", "aqa", "vision-only"]):
                    discovered.append(clean_name)
    except Exception:
        pass

    # Sort so most modern responsive / flash-lite models are prioritized first for speed
    def sort_key(name: str):
        score = 0
        if "3.5-flash-lite" in name:
            score += 50
        elif "3.1-flash-lite" in name:
            score += 40
        elif "flash-lite" in name:
            score += 30
        elif "3.8-flash" in name:
            score += 25
        elif "3.5-flash" in name:
            score += 20
        elif "flash" in name:
            score += 15
        return -score

    discovered.sort(key=sort_key)
    return discovered


# =====================================================================
# PLAN EXECUTION ON PANDAS DATAFRAME
# =====================================================================

def _resolve_column(candidate: Optional[str], df: pd.DataFrame) -> Optional[str]:
    if not candidate:
        return None
    c_clean = str(candidate).strip()
    if c_clean in df.columns:
        return c_clean

    col_map = {str(c).lower().strip(): c for c in df.columns}
    target = c_clean.lower()
    if target in col_map:
        return col_map[target]

    matches = difflib.get_close_matches(target, list(col_map.keys()), n=1, cutoff=0.6)
    if matches:
        return col_map[matches[0]]
    return None


def _fmt(val):
    if val is None:
        return "N/A"
    try:
        if pd.isna(val):
            return "N/A"
    except (TypeError, ValueError):
        pass
    if isinstance(val, (int, np.integer)):
        return f"{int(val):,}"
    if isinstance(val, (float, np.floating)):
        if abs(val) >= 1000:
            return f"{val:,.2f}"
        return f"{val:.4f}".rstrip("0").rstrip(".")
    return str(val)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def execute_plan(df: pd.DataFrame, plan: QueryPlan) -> dict:
    op = plan.operation

    if op == "row_count":
        return {"answer": f"The dataset has {len(df):,} total rows.", "type": "text", "plan": plan.model_dump()}

    if op == "column_count":
        return {"answer": f"The dataset contains {len(df.columns)} columns: {', '.join(str(c) for c in df.columns)}.", "type": "text", "plan": plan.model_dump()}

    if op == "data_types":
        types = "\n".join([f"• {c}: {df[c].dtype}" for c in df.columns])
        return {"answer": f"Dataset Column Types:\n{types}", "type": "text", "plan": plan.model_dump()}

    if op == "dataset_summary":
        return {
            "answer": f"Dataset Summary:\n• Rows: {len(df):,}\n• Columns: {len(df.columns)}\n• Total Missing: {int(df.isna().sum().sum()):,}\n• Duplicate Records: {int(df.duplicated().sum()):,}",
            "type": "text",
            "plan": plan.model_dump(),
        }

    if op == "missing":
        m_s = df.isna().sum()
        with_m = m_s[m_s > 0]
        if with_m.empty:
            return {"answer": f"All {len(df.columns)} columns have 0 missing values.", "type": "text", "plan": plan.model_dump()}
        lines = [f"• {c}: {v:,} missing ({v/len(df)*100:.1f}%)" for c, v in with_m.items()]
        return {"answer": f"Missing Values Detected ({int(with_m.sum()):,} total):\n" + "\n".join(lines), "type": "text", "plan": plan.model_dump()}

    if op == "duplicates":
        dups = int(df.duplicated().sum())
        return {"answer": f"Detected {dups:,} duplicate rows." if dups else "0 duplicate rows detected.", "type": "text", "plan": plan.model_dump()}

    if op == "unique":
        col = _resolve_column(plan.column, df) or df.columns[0]
        uniq = df[col].dropna().unique()
        return {"answer": f"Column '{col}' has {len(uniq):,} unique values. Sample: {', '.join(str(x) for x in uniq[:6])}.", "type": "text", "plan": plan.model_dump()}

    if op == "frequency":
        col = _resolve_column(plan.column, df) or df.columns[0]
        vc = df[col].value_counts().head(plan.n)
        lines = [f"• {k}: {v:,} times ({v/len(df)*100:.1f}%)" for k, v in vc.items()]
        return {"answer": f"Top frequent values in '{col}':\n" + "\n".join(lines), "type": "text", "plan": plan.model_dump()}

    if op == "aggregate":
        col = _resolve_column(plan.column, df) or df.select_dtypes(include=np.number).columns[0]
        series = pd.to_numeric(df[col], errors="coerce")
        agg = plan.aggregation or "mean"
        val = getattr(series, agg, series.mean)()
        return {"answer": f"The calculated {agg} of '{col}' is {_fmt(val)}.", "type": "text", "plan": plan.model_dump()}

    if op == "group_aggregate":
        grp = _resolve_column(plan.group_by, df) or df.select_dtypes(include=["object", "category"]).columns[0]
        col = _resolve_column(plan.column, df) or df.select_dtypes(include=np.number).columns[0]
        agg = plan.aggregation or "mean"

        grouped = df.groupby(grp)[col]
        agg_res = getattr(grouped, agg, grouped.mean)().sort_values(ascending=not plan.sort_desc).head(plan.n)
        lines = [f"• {k}: {_fmt(v)}" for k, v in agg_res.items()]
        top_k, top_v = agg_res.index[0], agg_res.iloc[0]
        return {
            "answer": f"Grouped {agg} of '{col}' by '{grp}'. Leading group is '{top_k}' with {_fmt(top_v)}:\n" + "\n".join(lines),
            "type": "text",
            "plan": plan.model_dump(),
        }

    if op in {"top_n", "bottom_n"}:
        col = _resolve_column(plan.column, df) or df.select_dtypes(include=np.number).columns[0]
        is_top = (op == "top_n")
        sorted_df = df.sort_values(by=col, ascending=not is_top).head(plan.n)
        return {
            "answer": f"Showing {'Top' if is_top else 'Bottom'} {plan.n} records sorted by '{col}'.",
            "type": "table",
            "data": _json_safe(sorted_df.to_dict(orient="records")),
            "plan": plan.model_dump(),
        }

    if op == "correlation":
        c1 = _resolve_column(plan.column, df) or df.select_dtypes(include=np.number).columns[0]
        c2 = _resolve_column(plan.second_column, df) or df.select_dtypes(include=np.number).columns[1]
        valid = df[[c1, c2]].dropna()
        r = float(pd.to_numeric(valid[c1], errors="coerce").corr(pd.to_numeric(valid[c2], errors="coerce")))
        strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
        direction = "positive" if r >= 0 else "negative"
        return {
            "answer": f"The Pearson correlation between '{c1}' and '{c2}' is {_fmt(r)} (indicating a {strength} {direction} relationship).",
            "type": "text",
            "plan": plan.model_dump(),
        }

    raise ValueError(f"Unknown operation: {op}")


# =====================================================================
# LOCAL QUERY PARSER (Offline Fallback)
# =====================================================================

def local_nl_query_engine(df: pd.DataFrame, question: str) -> QueryPlan:
    q = question.lower().strip()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    matched_num = [c for c in num_cols if str(c).lower() in q]
    matched_cat = [c for c in cat_cols if str(c).lower() in q]

    if any(k in q for k in ["row count", "how many rows", "total rows"]):
        return QueryPlan(operation="row_count")
    if any(k in q for k in ["columns", "what columns", "column count"]):
        return QueryPlan(operation="column_count")
    if any(k in q for k in ["missing", "null", "nan"]):
        return QueryPlan(operation="missing")
    if any(k in q for k in ["duplicate"]):
        return QueryPlan(operation="duplicates")

    if any(k in q for k in ["correlation", "correlated", "relationship"]):
        col1 = matched_num[0] if len(matched_num) >= 1 else (num_cols[0] if num_cols else None)
        col2 = matched_num[1] if len(matched_num) >= 2 else (num_cols[1] if len(num_cols) > 1 else None)
        return QueryPlan(operation="correlation", column=col1, second_column=col2)

    if any(k in q for k in [" by ", " per ", " each "]) or (matched_cat and matched_num):
        grp = matched_cat[0] if matched_cat else (cat_cols[0] if cat_cols else None)
        num = matched_num[0] if matched_num else (num_cols[0] if num_cols else None)
        agg = "sum" if any(k in q for k in ["sum", "total"]) else "max" if "highest" in q else "mean"
        return QueryPlan(operation="group_aggregate", column=num, group_by=grp, aggregation=agg, n=5)

    if any(k in q for k in ["top", "highest", "most expensive", "best"]):
        return QueryPlan(operation="top_n", column=matched_num[0] if matched_num else None, n=5)
    if any(k in q for k in ["bottom", "lowest", "cheapest", "least"]):
        return QueryPlan(operation="bottom_n", column=matched_num[0] if matched_num else None, n=5)

    if any(k in q for k in ["average", "mean", "total", "sum", "median"]):
        agg = "sum" if any(k in q for k in ["sum", "total"]) else "median" if "median" in q else "mean"
        return QueryPlan(operation="aggregate", column=matched_num[0] if matched_num else None, aggregation=agg)

    if matched_cat:
        return QueryPlan(operation="frequency", column=matched_cat[0], n=5)

    return QueryPlan(operation="dataset_summary")


# =====================================================================
# MAIN ANSWER QUESTION ENTRYPOINT
# =====================================================================

def answer_question(df: pd.DataFrame, question: str) -> dict:
    if not question or not question.strip():
        return {"answer": "Please ask a question about your dataset.", "type": "text"}

    q = question.strip()
    client = _get_gemini_client()

    # If Gemini is configured, discover all available Gemini models dynamically
    if client is not None:
        try:
            from google.genai import types

            discovered_models = discover_gemini_models(client)
            if discovered_models:
                prompt = f"""You are a Natural Language Data Query Agent.
Translate this user question into ONE safe QueryPlan JSON object for the dataset.
Do not invent columns. Use exact schema names.

COLUMNS: {json.dumps([str(c) for c in df.columns])}
NUMERIC: {json.dumps([str(c) for c in df.select_dtypes(include=np.number).columns])}
SAMPLE ROWS: {json.dumps(df.head(4).to_dict(orient="records"), default=str)}

USER QUESTION: {q}
"""
                # Attempt discovered models dynamically without hardcoding
                attempted = []
                successful = []

                for model_name in discovered_models[:3]:  # Attempt discovered models
                    attempted.append(model_name)
                    try:
                        resp = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=QueryPlan,
                                temperature=0,
                                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                            ),
                        )
                        parsed = getattr(resp, "parsed", None)
                        if parsed is not None:
                            plan = parsed if isinstance(parsed, QueryPlan) else QueryPlan.model_validate(parsed)
                        else:
                            clean_t = resp.text.strip().strip("`").removeprefix("json").strip()
                            plan = QueryPlan.model_validate_json(clean_t)

                        # Test against actual dataframe
                        res = execute_plan(df, plan)
                        successful.append((model_name, plan, res))
                        # We stop at the first successful model to respect API quotas
                        break
                    except Exception:
                        continue

                if successful:
                    winner_model, winner_plan, res = successful[0]
                    res.update({
                        "engine": "Google Gemini Dynamic Multi-Model AI",
                        "model": winner_model,
                        "discovered_models": discovered_models,
                        "total_discovered": len(discovered_models),
                        "question": q,
                    })
                    return res
        except Exception:
            pass  # Fall back to local query engine

    # Fallback to Local Smart Query Engine
    local_plan = local_nl_query_engine(df, q)
    res = execute_plan(df, local_plan)
    res.update({
        "engine": "Local Smart Analytics Engine (Offline / Safe)",
        "model": "rule-based semantic parser",
        "question": q,
    })
    return res
