import pandas as pd
import numpy as np


def create_pass_variable(df):
    """
    Create binary Pass variable from Total Mark.
    Pass = 1 if Total Mark >= 50
    Fail = 0 otherwise
    """

    df = df.copy()

    df["Total Mark"] = pd.to_numeric(
        df["Total Mark"],
        errors="coerce"
    )

    df["Pass"] = (
        df["Total Mark"] >= 50
    ).astype(int)

    return df


def prepare_data(df):
    """
    Prepare features for Logistic Regression.

    Predictors:
        - Course Code
        - Test 1
        - Test 2

    Outcome:
        - Pass
    """
    X = pd.get_dummies(
        df[
            [
                "Test 1",
                "Test 2"
            ]
        ],
        drop_first=False
    )    

    y = df["Pass"]

    return X, y


def calculate_exam_requirements(test1_pct, test2_pct):
    """
    Calculate the exam mark required to achieve
    various overall unit grades.

    Assessment structure:
        Test 1 = 20%
        Test 2 = 20%
        Exam   = 60%

    Inputs should be percentages (0-100).

    Example:
        test1_pct = 65
        test2_pct = 48
    """

    current_contribution = (
        0.20 * test1_pct
        + 0.20 * test2_pct
    )

    grade_targets = {
        "Pass (≥ 50%)": 50,
        "Credit (≥ 60%)": 60,
        "Distinction (≥ 70%)": 70,
        "High Distinction (≥ 80%)": 80
    }

    results = {}

    for label, target in grade_targets.items():

        exam_needed = (
            target - current_contribution
        ) / 0.60

        exam_needed = round(exam_needed, 1)

        if exam_needed < 0:
            exam_needed = 0.0

        results[label] = exam_needed

    return results