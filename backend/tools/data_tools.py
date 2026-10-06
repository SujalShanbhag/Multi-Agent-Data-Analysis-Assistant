from pathlib import Path
import pandas as pd
import numpy as np

ALLOWED_SUFFIXES = {".csv", ".xlsx", ".xls"}

def load_dataframe(path: str | Path) -> pd.DataFrame:
    """
    Load CSV or Excel datasets with automatic encoding detection.
    Cycles through common encodings to avoid UnicodeDecodeError.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Only CSV (.csv) and Excel (.xlsx, .xls) files are supported.")

    if suffix == ".csv":
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "iso-8859-1"]
        last_error = None
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except (UnicodeDecodeError, UnicodeError) as err:
                last_error = err
                continue
            except Exception:
                raise
        # Final fallback
        try:
            return pd.read_csv(path, encoding="latin-1", errors="replace")
        except Exception:
            raise last_error or ValueError(f"Could not decode CSV file: {path}")

    return pd.read_excel(path)


def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Generate structural overview of the dataset.
    """
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    dates = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    missing = df.isna().sum()

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "numeric_columns": [str(c) for c in numeric],
        "categorical_columns": [str(c) for c in categorical],
        "date_columns": [str(c) for c in dates],
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {str(k): int(v) for k, v in missing.items() if v > 0},
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024**2)), 3),
        "unique_by_column": {str(c): int(df[c].nunique(dropna=True)) for c in df.columns},
    }


def cleaning_report(df: pd.DataFrame) -> dict:
    """
    Produce cleaning recommendations for missing values and duplicates.
    """
    actions = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                strategy = "median imputation"
            else:
                strategy = "mode imputation"
            actions.append({
                "column": str(col),
                "issue": "missing values",
                "count": missing,
                "recommended": strategy,
            })

    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        actions.append({
            "issue": "duplicate rows",
            "count": duplicates,
            "recommended": "remove duplicate records",
        })

    return {
        "actions": actions,
        "duplicate_count": duplicates,
        "columns_with_missing": len([a for a in actions if a.get("column")]),
    }


def cleaned_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply automated deduplication and median/mode imputation.
    """
    out = df.copy()
    out = out.drop_duplicates()
    for col in out.columns:
        if out[col].isna().any():
            if pd.api.types.is_numeric_dtype(out[col]):
                median_val = out[col].median()
                if pd.notna(median_val):
                    out[col] = out[col].fillna(median_val)
            else:
                mode_vals = out[col].mode(dropna=True)
                if len(mode_vals) > 0:
                    out[col] = out[col].fillna(mode_vals.iloc[0])
    return out


def numeric_summary(df: pd.DataFrame) -> list:
    """
    Compute summary statistics for all numeric fields.
    """
    n = df.select_dtypes(include=np.number)
    if n.empty:
        return []
    desc = n.describe().T.reset_index().rename(columns={"index": "column"})
    return desc.round(4).to_dict(orient="records")


def categorical_summary(df: pd.DataFrame) -> list:
    """
    Compute top category frequencies for non-numeric fields.
    """
    result = []
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False).head(10)
        result.append({
            "column": str(col),
            "top_values": [{"value": str(k), "count": int(v)} for k, v in vc.items()]
        })
    return result
