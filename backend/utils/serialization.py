import math
import numpy as np
import pandas as pd

def clean_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def records_to_json(df, limit=100):
    return [
        {str(k): clean_value(v) for k, v in row.items()}
        for row in df.head(limit).to_dict(orient="records")
    ]
