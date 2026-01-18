# 📊 Customer Churn Prediction System (End-to-End ML + MLOps)

An industry-style, end-to-end Machine Learning system that predicts customer churn and delivers actionable business insights.
This project goes beyond a notebook—it implements a **production-grade ML pipeline** with experiment tracking, automated preprocessing, model comparison, and a live prediction application using Streamlit.

---

## 📑 Table of Contents

1. [Business Problem](#business-problem)
2. [Solution Overview](#solution-overview)
3. [System Architecture](#system-architecture)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [ML Pipeline](#ml-pipeline)
8. [Model Training &amp; Evaluation](#model-training--evaluation)
9. [Model Selection](#model-selection)
10. [Live Prediction App](#live-prediction-app)
11. [How to Run Locally](#how-to-run-locally)
12. [Key Learnings](#key-learnings)

---

## Business Problem

Customer churn directly impacts revenue.The objective of this system is to:

> Identify customers at high risk of churn and provide a data-driven foundation for proactive retention strategies.

The solution must:

- Detect churn patterns
- Quantify risk
- Train predictive models
- Serve predictions in real time
- Be reproducible and production-oriented

---

## Solution Overview

This project implements a **full ML lifecycle**:

- Business-driven EDA
- Clean data contract
- Automated preprocessing
- Model training & evaluation
- Experiment tracking with MLflow
- Business-based model selection
- Real-time inference via Streamlit

It bridges **Data Analytics + Data Science + MLOps principles** in one cohesive system.

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
```

---


## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- MLflow (Experiment Tracking)
- Streamlit (Application UI)
- Matplotlib, Seaborn
- Joblib

---



## Project Structure

```
customer-churn-mlops/
│
├── data/
│   └── processed/
│       └── churn_clean.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── ingestion.py
│   ├── validation.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── app/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```


---



## Exploratory Data Analysis

The EDA notebook:

- Audits data quality
- Handles missing values and duplicates
- Analyzes churn drivers
- Engineers business features
- Produces a clean analytical dataset

Key insights include:

- Month-to-month contracts show highest churn
- Early-tenure customers are most vulnerable
- Payment method strongly correlates with churn
- Revenue risk is concentrated in early lifecycle

This notebook acts as the **design document** for the ML system.

---



## ML Pipeline

The production pipeline is modular and reproducible:

1. **Ingestion** – Load clean dataset
2. **Preprocessing** –
   - Feature/target split
   - Scaling & encoding
   - Stratified train-test split
   - Class balancing (SMOTE)
3. **Training** – Train multiple models
4. **Evaluation** – Business-relevant metrics
5. **Artifact Creation** – Save model + preprocessor
6. **Inference** – Predict on new customers

---

## Model Training & Evaluation

Two models were trained:

| Model               | Accuracy | Precision | Recall | ROC-AUC |
| ------------------- | -------- | --------- | ------ | ------- |
| Logistic Regression | 0.74     | 0.50      | 0.78   | 0.85    |
| Random Forest       | 0.76     | 0.56      | 0.56   | 0.82    |

All experiments are tracked in **MLflow**, enabling:

- Parameter logging
- Metric comparison
- Artifact management
- Iterative experimentation

---

## Model Selection

Although Random Forest achieved slightly higher accuracy,**Logistic Regression was selected as the production model** because:

- It captures **78% of churners** (higher recall)
- It has better **ROC-AUC**
- It aligns with the business goal:
  > *“Do not miss customers who are about to leave.”*
  >

This reflects real-world ML decision-making.

---

## Live Prediction App

A Streamlit application provides:

- Interactive customer input
- Real-time churn prediction
- Probability + risk classification
- Business-friendly output

Run:

```bash
streamlit run app/app.py
```

The app demonstrates how the ML system is consumed by end users.


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

## Key Learnings

* Translating business questions into ML systems
* Designing reproducible pipelines
* Handling class imbalance in real data
* Experiment tracking and model comparison
* Schema parity between training and inference
* Building an ML-powered application
* Applying MLOps principles at entry level
