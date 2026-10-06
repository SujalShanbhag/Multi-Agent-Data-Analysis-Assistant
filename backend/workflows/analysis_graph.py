from backend.agents import (
    profiling_agent,
    cleaning_agent,
    eda_agent,
    statistics_agent,
    ml_agent,
    validation_agent,
    insight_agent,
    report_agent,
    gemini_conclusion_agent,
)

def run_workflow(df, dataset_name="dataset", target=None):
    """
    Coordinates end-to-end multi-agent dataset analysis workflow.
    Executes profiling, cleaning, EDA with visual charts, statistical tests,
    the complete 14+ ML model leaderboard, Gemini AI analysed conclusion,
    and PDF report generation.
    """
    # 1. Profiling
    profile = profiling_agent.run(df)

    # 2. Cleaning & preprocessing
    cleaning, cleaned = cleaning_agent.run(df)

    # 3. Exploratory Data Analysis & All Visualizations
    eda = eda_agent.run(cleaned)
    eda["outliers"] = eda_agent.outliers(cleaned)

    # 4. Statistical Tests
    statistics = statistics_agent.run(cleaned)

    # 5. Machine Learning (14+ Models Suite)
    ml = ml_agent.run(cleaned, target)

    # 6. Quality Validation
    validation = validation_agent.run(profile, eda, statistics, ml)

    # 7. Insights
    insights = insight_agent.run(profile, cleaning, eda, statistics, ml)

    # 8. Gemini AI Analysed Conclusion (Synthesizes complete findings)
    conclusion = gemini_conclusion_agent.generate_conclusion(
        dataset_name, profile, cleaning, eda, statistics, ml
    )

    # 9. PDF Report Generation (with Gemini Conclusion at the end)
    report_pdf_path = report_agent.create_report(
        dataset_name, profile, cleaning, eda, statistics, ml, validation, insights, conclusion
    )

    return {
        "profile": profile,
        "cleaning": cleaning,
        "eda": eda,
        "statistics": statistics,
        "ml": ml,
        "validation": validation,
        "insights": insights,
        "conclusion": conclusion,
        "report": report_pdf_path,
    }
