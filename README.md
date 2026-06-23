# POE-FINAL-SUBMISSION
# Sentiment Analytics & Corporate Vulnerability Triage System
* **Target Audience:** Executive Board of Directors, Medical Aid Scheme

---

## Executive Overview
This repository hosts the end-to-end Natural Language Processing (NLP) and supervised machine learning pipeline engineered to address an increasing volume of negative customer reviews on public consumer channels (primarily Hello Peter). 

The framework is structurally split into two core deliverables:
1. **Thematic Failure Extraction:** Isolating and categorizing unstructured customer friction vectors to find systemic operational flaws.
2. **Automated Risk Triage Pipeline:** Building and calibrating predictive text classification algorithms to flag and escalate critical incoming reviews in real-time.

---

##  Repository Components

| File Name | Format | Description |
| :--- | :--- | :--- |
| `medical_aid_text_analysis_report.pdf` | `PDF Document` | Board-ready executive report detailing business insights, strategic model rationales, comprehensive charts, and confusion matrices. |
| `medical_aid_text_analysis.ipynb` | `Jupyter Notebook` | Fully commented interactive core script containing cleaning routines, feature selections, and model optimization steps. |
| `hospital.csv` | `Comma-Separated Values` | Cleaned production database comprising 996 rows of verified patient feedback, star scores, and binary labels. |

---

## Technical Workflow & Implementation Summary

### 1. Data Integrity Profiling & EDA
* Verified zero missing or corrupted `NaN` observations across text strings and target labels.
* Calculated text distribution metrics (word count averages) and established quantitative validation via cross-tabulations (`pd.crosstab`) between explicit numeric star ratings (1 to 5) and binary sentiment flags.

### 2. Algorithmic Feature Selection
* Cleaned corpus by normalizing character arrays (lowercasing and regex filtering of non-alphabetic variables).
* Applied Term Frequency-Inverse Document Frequency (**TF-IDF**) parsing with boundary filters (`min_df=2`, `max_df=0.95`) to drop high-frequency noise and extract the top **500 highly discriminative unigram features**.

### 3. Model Development & Evaluation
* Used a **20% Stratified Validation Split** to ensure matching class frequencies across training and test subsets.
* Implemented a linear model baseline (**Logistic Regression**) alongside non-linear decision networks (**Random Forest Classifier**) to benchmark high-dimensional sparse vector spaces.

### 4. Shortcoming Diagnosis & Hyperparameter Tuning
* **Identified Shortcoming:** Initial benchmark evaluations suffered from low Recall metrics on negative feedback rows, which introduces enterprise risk (allowing critical customer complaints to slip through unflagged).
* **Technical Solution:** Retrained the champion linear architecture by scaling text regularization properties ($C \rightarrow 2.5$) and injecting automated inverse penalty adjustments (`class_weight='balanced'`).
* **Outcome:** Successfully achieved an overall out-of-sample accuracy of **95.00%**, significantly driving down false negative occurrences.

---

## Key Corporate Areas of Concern Discovered

By isolating text features exclusively linked to low-satisfaction thresholds (Ratings $\le$ 2), the system programmatically flagged three critical administrative operational failure points:
1. **OPD Scheduling & Queue Management:** Prolonged customer waiting rooms indicated by top keywords like *wait, time, queue, delay*.
2. **Financial Transparency & Pricing Stress:** Brand value friction points caused by unexpected expenses, tracked via keywords like *money, expensive, charges, pocket*.
3. **Diagnostic Trust & Authenticity:** Customer anxiety regarding report contradictions or repetitive test validation requests, isolated via terms like *test, reports, diagnosis, wrong*.

---

## Setup, Installation & Reproduction

### Environmental Prerequisites
Ensure your local Python installation contains the fundamental analytical dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
