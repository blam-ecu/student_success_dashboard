# Beyond Prediction: Student-Facing Learning Analytics Dashboard

A student-facing learning analytics dashboard that combines:

- Historical passing likelihood estimation
- Examination-planning support
- Temporal model validation
- Sigmoid-calibrated logistic regression

The project was developed to support academic-risk awareness and examination planning using only early assessment performance.

---

# Research Overview

This project accompanies the paper:

**Beyond Prediction: A Student-Facing Learning Analytics Dashboard for Academic Risk Awareness and Examination Planning**

The dashboard uses a parsimonious logistic regression model based on:

- Test 1
- Test 2

to estimate historical passing likelihood and provide examination-planning guidance.

The final deployed model is a **sigmoid-calibrated logistic regression model** validated using an independent temporal holdout cohort.

---

# Project Structure

```text
student_success/
│
├── train_model.py
├── temporal_validation.py
├── dashboard.py
├── utils.py
│
├── data/
│   ├── training.xlsx
│   └── temporal_holdout.xlsx
│
├── models/
│   ├── logistic_model.pkl
│   ├── feature_columns.pkl
│   └── course_list.pkl
│
├── outputs/
│   ├── model_comparison.xlsx
│   ├── reliability_comparison.png
│   └── dashboard_screenshot.png
│
└── requirements.txt# student_success_dashboard
