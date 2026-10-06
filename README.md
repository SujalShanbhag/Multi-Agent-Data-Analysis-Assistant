# Multi-Agent Data Analysis Assistant

A web-based AI data analysis system that uses multiple specialized agents to automatically understand, clean, analyze, visualize, and generate insights from CSV and Excel datasets.

The system can work in two modes:

1. **With Google Gemini API** – provides AI-powered natural language understanding.
2. **Without Gemini API** – uses the built-in local analysis engine.

A Gemini API key is optional.

---

## 1. Project Overview

The Multi-Agent Data Analysis Assistant is designed to automate the process of analyzing tabular datasets.

Instead of performing every analysis manually, different agents are responsible for different tasks. A Coordinator Agent manages these agents and combines their results into a complete analysis.

The user can upload a CSV or Excel file and perform:

- Dataset profiling
- Data cleaning
- Exploratory Data Analysis (EDA)
- Statistical analysis
- Machine learning
- Anomaly detection
- Data validation
- Natural language querying
- Automatic insights
- Report generation

The system provides the results through a web-based dashboard.

---

## 2. How the System Works

The general workflow is:

```text
User
  |
  v
Upload CSV / Excel Dataset
  |
  v
Coordinator Agent
  |
  +--> Profiling Agent
  |
  +--> Cleaning Agent
  |
  +--> EDA Agent
  |
  +--> Statistical Agent
  |
  +--> Machine Learning Agent
  |
  +--> Validation Agent
  |
  +--> Insight Agent
  |
  +--> Report Agent
  |
  +--> Natural Language Query Agent
  |
  v
Final Analysis
  |
  v
Dashboard / Report
```

The Coordinator Agent manages the overall workflow and ensures that the different agents work on the same dataset.

---

## 3. Multi-Agent System

### 3.1 Coordinator Agent

The Coordinator Agent manages the complete analysis process.

**Responsibilities:**

- Receives the uploaded dataset
- Maintains the current dataset
- Coordinates different agents
- Controls the analysis workflow
- Combines the results from different agents

---

### 3.2 Data Profiling Agent

The Profiling Agent examines the basic structure and quality of the dataset.

**It provides:**

- Number of rows
- Number of columns
- Column names
- Data types
- Missing values
- Missing-value percentages
- Unique values
- Basic dataset information

---

### 3.3 Data Cleaning Agent

The Cleaning Agent identifies common data quality problems.

**It can handle:**

- Missing values
- Duplicate rows
- Numerical data cleaning
- Categorical data cleaning
- Median-based numerical imputation
- Mode-based categorical imputation

The agent prepares the dataset for further analysis.

---

### 3.4 EDA Agent

The Exploratory Data Analysis Agent studies the characteristics of the dataset.

**It generates:**

- Descriptive statistics
- Minimum values
- Maximum values
- Mean
- Median
- Standard deviation
- Data distributions
- Numerical relationships
- IQR-based outlier information

---

### 3.5 Statistical Analysis Agent

The Statistical Agent performs statistical analysis on suitable numerical columns.

**It includes:**

- Pearson correlation
- Correlation matrices
- Statistical summaries
- Shapiro-Wilk normality testing where applicable

The results help identify relationships and patterns between variables.

---

### 3.6 Machine Learning Agent

The Machine Learning Agent automatically performs machine learning when the dataset is suitable.

Depending on the selected target and data type, it can perform:

- Classification
- Regression
- Unsupervised clustering

**Classification metrics:**

- Accuracy
- F1 Score

**Regression metrics:**

- R² Score
- RMSE

For clustering, the system uses K-Means.

The system also performs preprocessing before model training.

If the machine-learning library cannot be loaded on a particular system, the rest of the application can continue running without stopping the complete application.

---

### 3.7 Validation Agent

The Validation Agent checks whether the analysis results are valid and whether the dataset satisfies basic requirements.

**It checks for problems such as:**

- Empty datasets
- Invalid data
- Missing required information
- Invalid analysis results
- Machine-learning execution problems

This helps improve the reliability of the system.

---

### 3.8 Insight Agent

The Insight Agent converts analysis results into understandable findings.

**It identifies useful information such as:**

- Important patterns
- Strong relationships
- Data quality issues
- Important numerical trends
- Possible business findings

The purpose is to make technical analysis easier to understand.

---

### 3.9 Report Agent

The Report Agent generates an HTML analysis report.

**The report can contain:**

- Dataset information
- Data quality results
- Statistical analysis
- EDA results
- Machine-learning results
- Insights
- Validation information

The generated report can be opened directly in a web browser.

---

### 3.10 Natural Language Query Agent

The Natural Language Query Agent allows users to ask questions about their uploaded dataset using normal language.

For example, a user can ask:

- Which product has the highest sales?
- What is the average price?
- Are there missing values?
- Which category has the most records?
- What is the relationship between two numerical columns?

The system does not depend on a fixed list of questions.

It dynamically uses the columns and information available in the uploaded dataset.

Dataset-specific questions and column names are not hardcoded into the application.

---

## 4. Gemini AI Integration

The Natural Language Query Agent can use Google Gemini AI to understand user questions.

Gemini helps convert a natural-language question into an appropriate analysis operation.

For example:

**User question:**

```text
Which fruit generated the highest total sales?
```

Gemini can understand that the required operation is:

1. Identify the fruit column.
2. Identify the sales column.
3. Group the data by fruit.
4. Calculate total sales for each fruit.
5. Compare the totals.
6. Return the fruit with the highest total sales.

The actual calculation is performed by the application using the dataset.

This means Gemini is used for understanding the question, while the application performs the data operation.

---

## 5. Gemini API Is Optional

The application works both with and without a Gemini API key.

### 5.1 Without Gemini API

The application uses its built-in local analysis engine.

It can still perform:

- Dataset profiling
- Data cleaning
- EDA
- Statistical analysis
- Machine learning
- Validation
- Insights
- Reports
- Basic natural-language dataset queries

No Gemini API key is required for the local analysis features.

---

### 5.2 With Gemini API

When a Gemini API key is configured, the Natural Language Query Agent can use Gemini for more flexible natural-language understanding.

The application can dynamically discover compatible Gemini models instead of depending on only one fixed model.

If one available Gemini model is unavailable or fails, the system can try other compatible models.

This makes the Gemini integration more flexible and reduces dependence on a single model.

---

## 6. Dataset Support

The application supports:

- CSV files
- Excel XLSX files
- Excel XLS files

CSV files can be loaded using multiple common encodings:

- UTF-8
- UTF-8-SIG
- CP1252
- Latin-1

This helps prevent common CSV encoding errors.

The system does not require a fixed dataset structure.

Different datasets can contain different:

- Column names
- Number of rows
- Number of columns
- Numerical columns
- Categorical columns
- Date columns

The analysis adapts to the uploaded dataset.

---

## 7. Dashboard

The project includes a modern dark dashboard.

The dashboard provides:

- Dataset upload
- Dataset information
- Analysis controls
- Agent status
- Analysis results
- Natural-language query interface
- Generated insights
- Report access

The interface is designed to make a multi-agent data-analysis system easy to use without requiring the user to write Python code.

---

## 8. Project Structure

```text
Multi-Agent-Data-Analysis-Assistant/
│
├── backend/
│   ├── main.py
│   │
│   ├── agents/
│   │   ├── profiling_agent.py
│   │   ├── cleaning_agent.py
│   │   ├── eda_agent.py
│   │   ├── statistics_agent.py
│   │   ├── ml_agent.py
│   │   ├── validation_agent.py
│   │   ├── insight_agent.py
│   │   └── report_agent.py
│   │
│   ├── tools/
│   │   └── data_tools.py
│   │
│   ├── workflows/
│   │   └── analysis_graph.py
│   │
│   └── utils/
│       └── serialization.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── data/
│   └── sample_sales.csv
│
├── reports/
│
├── tests/
│   └── test_core.py
│
├── requirements.txt
├── .env.example
├── README.md
└── .gitignore
```

---

## 9. Technologies Used

### Frontend

- HTML
- CSS
- JavaScript

### Backend

- Python
- FastAPI
- Uvicorn

### Data Analysis

- Pandas
- NumPy
- SciPy

### Machine Learning

- Scikit-learn

### Visualization

- Matplotlib
- Plotly

### AI

- Google Gemini API
- Google GenAI Python SDK

### File Processing

- OpenPyXL

---

## 10. Requirements

Before running the project, make sure you have:

- Python 3.10 or newer
- Windows, Linux, or macOS

A Gemini API key is optional.

An internet connection is required when using Gemini AI features.

---

## 11. Installation

Open PowerShell inside the project folder.

### Step 1: Create a virtual environment

```powershell
python -m venv .venv
```

### Step 2: Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## 12. Gemini API Setup

Gemini is optional.

If you want to use Gemini AI, create a `.env` file from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Open the `.env` file and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

If you do not provide a Gemini API key, the application will continue working using its local analysis engine.

---

## 13. Running the Application

From the project root directory, run:

```powershell
python -m backend.main
```

Then open your browser and go to:

```text
http://127.0.0.1:8000
```

---

## 14. Using the Application

### Step 1: Upload Dataset

Upload a CSV or Excel file through the dashboard.

### Step 2: Dataset Profiling

The system automatically examines the dataset and displays its basic information.

### Step 3: Run Analysis

Start the complete analysis.

The different agents work on different parts of the dataset.

### Step 4: View Results

The dashboard displays:

- Data profile
- Cleaning results
- EDA results
- Statistical results
- Machine-learning results
- Validation results
- Insights

### Step 5: Ask Questions

Use the natural-language query section to ask questions about the uploaded dataset.

If Gemini is configured, Gemini helps understand the question.

If Gemini is not configured, the local query engine is used.

### Step 6: Generate Report

After analysis, an HTML report can be generated and opened from the application.

---

## 15. Natural Language Query Working

The query system follows this process:

```text
User Question
      |
      v
Natural Language Query Agent
      |
      +--------------------------+
      |                          |
      v                          v
Gemini API Available?       Gemini API Not Available
      |                          |
      v                          v
Gemini Model                Local Query Engine
      |                          |
      +------------+-------------+
                   |
                   v
             Analysis Plan
                   |
                   v
            Dataset Analysis
                   |
                   v
                Result
```

The system uses the actual dataset structure when processing questions.

It does not require questions to be manually added to the code.

It also does not require dataset-specific column names to be hardcoded.

---

## 16. API Endpoints

The backend provides the following endpoints:

```text
GET  /api/health
GET  /api/sample
POST /api/upload
POST /api/analyze
POST /api/query
GET  /api/report
```

The frontend uses these endpoints to communicate with the backend.

---

## 17. Testing

Run the automated tests from the project root:

```powershell
python -m pytest
```

The tests verify important parts of the application and help identify problems before deployment.

---

## 18. Advantages of the Project

- Multi-agent architecture
- Automatic dataset analysis
- Supports CSV and Excel
- Works with different dataset structures
- Gemini AI integration
- Works without Gemini API
- Natural-language data queries
- Automatic machine learning
- Statistical analysis
- Data validation
- Automated insights
- HTML report generation
- Modern web dashboard
- Modular architecture
- Easy to extend with additional agents

---

## 19. Project Objective

The main objective of this project is to develop an intelligent multi-agent system that can automatically analyze tabular data and assist users in understanding their datasets without requiring them to manually perform every data-analysis operation.

The system combines traditional data-analysis techniques with AI-powered natural-language understanding to provide an interactive and automated data-analysis experience.