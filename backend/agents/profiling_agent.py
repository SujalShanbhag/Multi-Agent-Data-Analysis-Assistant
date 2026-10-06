from backend.tools.data_tools import profile_dataframe, numeric_summary, categorical_summary

def run(df):
    profile = profile_dataframe(df)
    profile["numeric_summary"] = numeric_summary(df)
    profile["categorical_summary"] = categorical_summary(df)
    return profile
