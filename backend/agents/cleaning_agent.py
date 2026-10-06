from backend.tools.data_tools import cleaning_report, cleaned_dataframe

def run(df):
    report = cleaning_report(df)
    cleaned = cleaned_dataframe(df)
    return report, cleaned
