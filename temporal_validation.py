import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix
)

from sklearn.calibration import calibration_curve

from utils import (
    create_pass_variable,
    prepare_data
)

# --------------------------------------------------
# Expected Calibration Error
# --------------------------------------------------

def expected_calibration_error(
    y_true,
    probs,
    n_bins=10
):

    bins = np.linspace(
        0,
        1,
        n_bins + 1
    )

    binids = np.digitize(
        probs,
        bins
    ) - 1

    ece = 0

    for b in range(n_bins):

        mask = (
            binids == b
        )

        if np.sum(mask) > 0:

            accuracy = np.mean(
                y_true[mask]
            )

            confidence = np.mean(
                probs[mask]
            )

            ece += (
                np.abs(
                    accuracy - confidence
                )
                * np.sum(mask)
                / len(y_true)
            )

    return ece


# --------------------------------------------------
# Load Holdout Data
# --------------------------------------------------

df_holdout = pd.read_excel(
    "data/temporal_holdout 252.xlsx"
)


print(
    "\nHoldout dataset shape:"
)

print(
    df_holdout.shape
)

print(
    "\nColumns:"
)

print(
    df_holdout.columns.tolist()
)

df_holdout = create_pass_variable(
    df_holdout
)

print(
    "\nPass distribution:"
)

print(
    df_holdout["Pass"].value_counts()
)

print(
    df_holdout[
        [
            "Test 1",
            "Test 2"
        ]
    ].describe()
)

# --------------------------------------------------
# Prepare Data
# --------------------------------------------------

X_test, y_test = prepare_data(
    df_holdout
)

# --------------------------------------------------
# Models To Compare
# --------------------------------------------------

model_files = {
    "Raw LR":
        "models/logistic_model_raw.pkl",

    "Sigmoid":
        "models/logistic_model_sigmoid.pkl",

    "Isotonic":
        "models/logistic_model_isotonic.pkl"
}

# --------------------------------------------------
# Feature Columns
# --------------------------------------------------

with open(
    "models/feature_columns.pkl",
    "rb"
) as f:

    feature_columns = pickle.load(f)

X_test = X_test.reindex(
    columns=feature_columns,
    fill_value=0
)

# --------------------------------------------------
# Evaluate Models
# --------------------------------------------------

results = []

for model_name, model_file in model_files.items():

    print(
        f"\nEvaluating {model_name}"
    )

    with open(
        model_file,
        "rb"
    ) as f:

        model = pickle.load(f)

    probs = model.predict_proba(
        X_test
    )[:, 1]

    preds = (
        probs >= 0.50
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probs
    )

    pr_auc = average_precision_score(
        y_test,
        probs
    )

    accuracy = accuracy_score(
        y_test,
        preds
    )

    precision = precision_score(
        y_test,
        preds,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        preds,
        zero_division=0
    )

    brier = brier_score_loss(
        y_test,
        probs
    )

    ece = expected_calibration_error(
        np.array(y_test),
        probs
    )

    results.append(
        {
            "Model":
                model_name,

            "ROC-AUC":
                roc_auc,

            "PR-AUC":
                pr_auc,

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "Brier Score":
                brier,

            "ECE":
                ece
        }
    )

    # --------------------------------------
    # Generate Reliability Diagrams
    # --------------------------------------

    fraction_pos, mean_pred = calibration_curve(
        y_test,
        probs,
        n_bins=10,
        strategy="uniform"
    )

    plt.figure(
        figsize=(5, 5)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray"
    )

    plt.plot(
        mean_pred,
        fraction_pos,
        marker="o"
    )

    plt.xlabel(
        "Predicted Probability"
    )

    plt.ylabel(
        "Observed Probability"
    )

    plt.title(
        f"Reliability Diagram - {model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        f"outputs/reliability_{model_name.replace(' ','_')}.png",
        dpi=300
    )

    plt.close()

# --------------------------------------------------
# Combined Reliability Diagram Figure
# --------------------------------------------------

    img_sigmoid = mpimg.imread(
        "outputs/reliability_Sigmoid.png"
    )

    img_isotonic = mpimg.imread(
        "outputs/reliability_Isotonic.png"
    )

    img_raw = mpimg.imread(
        "outputs/reliability_Raw_LR.png"
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    images = [
        img_raw,
        img_isotonic,
        img_sigmoid
    ]

    titles = [
        "Raw Logistic Regression",
        "Isotonic Calibration",
        "Sigmoid Calibration"
    ]

    for ax, img, title in zip(
        axes,
        images,
        titles
    ):

        ax.imshow(img)
        ax.set_title(
            title,
            fontsize=12
        )
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        "outputs/reliability_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved: outputs/reliability_comparison.png"
    )    

# --------------------------------------------------
# Results Table
# --------------------------------------------------

results_df = pd.DataFrame(
    results
).round(4)

print("\n")
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)

# --------------------------------------------------
# Save Results
# --------------------------------------------------

results_df.to_excel(
    "outputs/model_comparison.xlsx",
    index=False
)

print(
    "\nSaved:"
)

print(
    "outputs/model_comparison.xlsx"
)