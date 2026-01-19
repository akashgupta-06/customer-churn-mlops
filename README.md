# 📊 Customer Churn Prediction System (End-to-End ML + MLOps + Analytics)

An industry-style, end-to-end Machine Learning and Analytics system that predicts customer churn and delivers actionable business insights.

This project goes beyond notebooks—it implements a **production-grade ML pipeline**, a **business analytics layer**, automated preprocessing, experiment tracking, CI, and a live prediction application.

---

## 📑 Table of Contents

1. [Business Problem](#business-problem)
2. [Solution Overview](#solution-overview)
3. [System Architecture](#system-architecture)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Business Analytics Layer](#business-analytics-layer)
7. [Exploratory Data Analysis](#exploratory-data-analysis)
8. [ML Pipeline](#ml-pipeline)
9. [Model Training &amp; Evaluation](#model-training--evaluation)
10. [Model Selection](#model-selection)
11. [Power BI Dashboard](#power-bi-dashboard)
12. [Live Prediction App](#live-prediction-app)
13. [CI/CD](#cicd)
14. [How to Run Locally](#how-to-run-locally)
15. [Key Insight](#key-insight)
16. [Final Recommendations](#final-recommendations)
17. [Author &amp; Contact](#author--contact)

## Business Problem

Customer churn directly impacts revenue. The objective of this system is to:

> Identify customers at high risk of churn and provide a data-driven foundation for proactive retention strategies.

The solution must:

- Detect churn patterns
- Quantify business risk
- Train predictive models
- Serve predictions in real time
- Be reproducible and production-oriented

---

## Solution Overview

This project implements a **full data-to-decision lifecycle**:

- Business-driven EDA
- Clean data contract
- Automated preprocessing
- Model training & evaluation
- Experiment tracking with MLflow
- Business-based model selection
- Analytics dashboard for stakeholders
- Real-time inference via Streamlit
- CI for pipeline reliability

It bridges **Data Analytics + Data Science + MLOps** in one cohesive system.

---

## System Architecture

```
     Raw Data
	↓
EDA & Business Insights (Notebook)
	↓
Clean Data Contract (churn_clean.csv)
	↓
    Ingestion
	↓
    Preprocessing
	↓ 
      Train
	↓
     Evaluate
	↓
MLflow (Experiment Tracking)
	↓
Production Model Artifact
	↓
   Predict Layer
	↓
Streamlit App (User Interface)
	↓
Power BI Dashboard (Business View)
```

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- MLflow (Experiment Tracking)
- Streamlit (Inference UI)
- Power BI (Business Dashboard)
- Matplotlib, Seaborn
- GitHub Actions (CI)

---

## Project Structure

```

customer-churn-mlops/
│
├── data/
│ ├── raw/
│ └── processed/
│ 	└── churn_clean.csv
│
├── notebooks/
│ ├── 01_churn_eda.ipynb
│ └── 02_churn_business_analytics.ipynb
│
├── dashboards/
│  	└── customer_churn_dashboard.pbix
│ 	└── dashboard_overview.png
│
├── src/
│ ├── ingestion.py
│ ├── validation.py
│ ├── preprocessing.py
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
│
├── app/
│ ├── app.py
│ └── streamlit_app.png
│
├── .github/workflows/ci.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Business Analytics Layer

A dedicated analytics layer translates data into business decisions:

- Executive KPIs (Churn %, Revenue at Risk)
- Segment analysis by:
  - Contract type
  - Tenure bucket
  - Payment method
- Revenue impact prioritization
- Actionable retention insights

Notebook: `notebooks/02_churn_business_analytics.ipynb`

This makes the project equally strong for **Data Analyst** roles.

---

## Exploratory Data Analysis

The EDA notebook:

- Audits data quality
- Handles missing values and duplicates
- Analyzes churn drivers
- Engineers business features
- Produces a clean analytical dataset

Key insights:

- Month-to-month contracts show highest churn
- Early-tenure customers are most vulnerable
- Payment method correlates strongly with churn
- Revenue risk is concentrated early in the lifecycle

---

## ML Pipeline

1. Ingestion – Load clean dataset
2. Preprocessing
   - Feature/target split
   - Encoding & scaling
   - Stratified train-test split
   - Class balancing (SMOTE)
3. Training – Multiple models
4. Evaluation – Business-relevant metrics
5. Artifact creation – Model + preprocessor
6. Inference – Predict on new customers

---

## Model Training & Evaluation

| Model               | Accuracy | Precision | Recall | ROC-AUC |
| ------------------- | -------- | --------- | ------ | ------- |
| Logistic Regression | 0.74     | 0.50      | 0.78   | 0.85    |
| Random Forest       | 0.76     | 0.56      | 0.56   | 0.82    |

All runs are tracked in **MLflow**.

---

## Model Selection

Logistic Regression was selected because:

- Captures **78% of churners** (higher recall)
- Better **ROC-AUC**
- Aligns with business goal:

> “Do not miss customers who are about to leave.”

---

## Power BI Dashboard

An executive dashboard provides:

- KPIs: Customers, Churn %, Revenue, Revenue at Risk
- Drivers: Contract, Tenure, Payment Method
- Interactive segmentation

![1768766469095](image/README/1768766469095.png)

---

## Live Prediction App

A Streamlit application provides:

- Structured customer input
- Real-time churn prediction
- Probability + risk classification
- Business-oriented recommendations

Run:

```bash
streamlit run app/app.py
```

---

## CI/CD

GitHub Actions runs on every push:

* Installs dependencies
* Executes preprocessing
* Trains model
* Validates pipeline health

This ensures:

* Reproducibility
* Early failure detection
* Production discipline

Workflow: `.github/workflows/ci.yml`

---

## How to Run Locally

1. Create a virtual environment
2. Install dependencies:

<pre class="overflow-visible! px-0!" data-start="4804" data-end="4847"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip install -r requirements.txt
</span></span></code></div></div></pre>

3. Train the model:

<pre class="overflow-visible! px-0!" data-start="4870" data-end="4901"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python src/train.py
</span></span></code></div></div></pre>

4. Evaluate:

<pre class="overflow-visible! px-0!" data-start="4917" data-end="4951"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python src/evaluate.py
</span></span></code></div></div></pre>

5. Launch the app:

<pre class="overflow-visible! px-0!" data-start="4973" data-end="5009"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>streamlit run app/app.py
</span></span></code></div></div></pre>

6. (Optional) View experiments:

<pre class="overflow-visible! px-0!" data-start="5044" data-end="5065"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>mlflow ui
</span></span></code></div></div></pre>

Open: `http://127.0.0.1:5000`

---

## Key Insights

The single most critical insight from this system is:

> **Month-to-month customers account for the highest churn (≈43%) and nearly 87% of total revenue at risk.**

This reveals that churn is not evenly distributed—it is **concentrated in early-tenure, low-commitment customers**.Retention efforts should therefore focus on:

- The first 6–12 months of the customer lifecycle
- Customers on month-to-month contracts
- High-risk payment method segments

This insight directly informs where business teams should invest retention budgets.

---

## Final Recommendations

Based on analytics and model outputs, the business should:

1. Launch targeted retention campaigns for **month-to-month customers**
2. Introduce onboarding and loyalty programs for customers in their **first 6 months**
3. Use churn probability to prioritize **high-risk, high-value customers**
4. Align retention spend with **revenue at risk**, not just churn count
5. Integrate this system into CRM workflows for real-time decision support

This transforms churn prediction from a technical model into a **revenue-protection engine**.

---

## Author & Contact

**Akash Gupta**Aspiring Data Analyst / Data Scientist

- Email: akashgupta010603@gmail.com
- LinkedIn: https://www.linkedin.com/in/akashgupta06
- GitHub: https://github.com/akashgupta-06

This project was built to demonstrate real-world data analytics, machine learning, and MLOps capabilities in a production-style system.
