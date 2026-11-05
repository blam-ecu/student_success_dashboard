# Install dependencies
# pip install -r requirements.txt

# Train models
# python train_model.py

# Run temporal validation
# python temporal_validation.py

# Launch dashboard
# streamlit run dashboard.py

import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils import (
    create_pass_variable,
    prepare_data
)

# -------------------------------------------
# Load Data
# -------------------------------------------

df = pd.read_excel(
    "data/training.xlsx"
)

# -------------------------------------------
# Create Pass Variable
# -------------------------------------------

df = create_pass_variable(df)

# -------------------------------------------
# Correlation Matrix
# -------------------------------------------

corr_df = df[
    [
        "Test 1",
        "Test 2"
        
    ]
].corr()

print("\nCorrelation Matrix")
print(corr_df)

print(df[["Test 1", "Test 2"]].describe())

print(
    pd.cut(df["Test 1"], bins=5).value_counts().sort_index()
)

print(
    pd.cut(df["Test 2"], bins=5).value_counts().sort_index()
)

df_check = df[
    (df["Test 1"] >= 9) &
    (df["Test 1"] <= 11) &
    (df["Test 2"] >= 9) &
    (df["Test 2"] <= 11)
]

print(df_check["Pass"].mean())

# -------------------------------------------
# VIF
# -------------------------------------------

vif_data = df[
    [
        "Test 1",
        "Test 2"
    ]
].copy()

import statsmodels.api as sm

vif_data = df[
    [
        "Test 1",
        "Test 2"
    ]
].copy()

vif_data = sm.add_constant(vif_data)

vif_results = pd.DataFrame()

vif_results["Variable"] = vif_data.columns

vif_results["VIF"] = [
    variance_inflation_factor(
        vif_data.values,
        i
    )
    for i in range(vif_data.shape[1])
]


# -------------------------------------------
# Prepare Data
# -------------------------------------------

X_train, y_train = prepare_data(df)

# -------------------------------------------
# Logistic Regression Models
# -------------------------------------------

from sklearn.calibration import CalibratedClassifierCV

# Raw Logistic Regression

raw_model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

raw_model.fit(
    X_train,
    y_train
)

# -------------------------------------------
# Coefficients from Raw Model
# -------------------------------------------

print("\nIntercept")
print(
    raw_model.named_steps["model"].intercept_
)

coef_df = pd.DataFrame(
    {
        "Feature": X_train.columns,
        "Coefficient": raw_model.named_steps["model"].coef_[0]
    }
)

print("\nTop positive coefficients")
print(
    coef_df.sort_values(
        "Coefficient",
        ascending=False
    ).head(10)
)

# -------------------------------------------
# Sigmoid Calibration
# -------------------------------------------

sigmoid_model = CalibratedClassifierCV(
    estimator=raw_model,
    method="sigmoid",
    cv=5
)

print("\nTraining SIGMOID model")

sigmoid_model.fit(
    X_train,
    y_train
)

# -------------------------------------------
# Isotonic Calibration
# -------------------------------------------

isotonic_model = CalibratedClassifierCV(
    estimator=raw_model,
    method="isotonic",
    cv=5
)

print("\nTraining ISOTONIC model")

isotonic_model.fit(
    X_train,
    y_train
)

# -------------------------------------------
# Save Models
# -------------------------------------------

with open(
    "models/logistic_model_raw.pkl",
    "wb"
) as f:

    pickle.dump(
        raw_model,
        f
    )

with open(
    "models/logistic_model_sigmoid.pkl",
    "wb"
) as f:

    pickle.dump(
        sigmoid_model,
        f
    )

with open(
    "models/logistic_model_isotonic.pkl",
    "wb"
) as f:

    pickle.dump(
        isotonic_model,
        f
    )

# -------------------------------------------
# Save Deployment Model
# -------------------------------------------

# Choose one for dashboard deployment

deployment_model = sigmoid_model

with open(
    "models/logistic_model.pkl",
    "wb"
) as f:

    pickle.dump(
        deployment_model,
        f
    )

# -------------------------------------------
# Save Feature Columns
# -------------------------------------------

with open(
    "models/feature_columns.pkl",
    "wb"
) as f:

    pickle.dump(
        list(X_train.columns),
        f
    )

# -------------------------------------------
# Save Course Codes
# -------------------------------------------

course_list = sorted(
    df["Course Code"]
    .dropna()
    .unique()
    .tolist()
)

with open(
    "models/course_list.pkl",
    "wb"
) as f:

    pickle.dump(
        course_list,
        f
    )

# -------------------------------------------
# Save Diagnostics
# -------------------------------------------

with pd.ExcelWriter(
    "outputs/model_diagnostics.xlsx",
    engine="openpyxl"
) as writer:

    corr_df.to_excel(
        writer,
        sheet_name="Correlation"
    )

    vif_results.to_excel(
        writer,
        sheet_name="VIF",
        index=False
    )

# -------------------------------------------
# Save Coefficients
# -------------------------------------------

coef_df.sort_values(
    "Coefficient",
    ascending=False,
    inplace=True
)

coef_df.to_excel(
    "outputs/logistic_coefficients.xlsx",
    index=False
)

# -------------------------------------------
# Output
# -------------------------------------------

print("\nModels saved:")

print("  logistic_model_raw.pkl")
print("  logistic_model_sigmoid.pkl")
print("  logistic_model_isotonic.pkl")

print(
    "\nCurrent dashboard deployment model:"
)

print(
    "  logistic_model.pkl -> sigmoid"
)