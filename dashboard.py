# Dashboard uses sigmoid-calibrated logistic regression

import pickle
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from utils import calculate_exam_requirements

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Beyond Prediction",
    page_icon="🎓",
    layout="wide"
)

# ==================================================
# LOAD MODEL
# ==================================================

with open("logistic_model_sigmoid.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# with open("models/course_list.pkl", "rb") as f:
#     course_list = pickle.load(f)

# ==================================================
# TITLE
# ==================================================

st.title("🎓 Beyond Prediction")

st.markdown("""
### Academic Risk Awareness and Success Planning

Use your Test 1 and Test 2 results to:
- Understand your current position
- Determine what to aim for in the final examination
- Estimate historical passing likelihood based on previous students with similar results
""")

# ==================================================
# SIDEBAR INPUTS
# ==================================================

st.sidebar.header("Student Inputs")

# course = st.sidebar.selectbox(
#     "Course Code",
#     course_list
# )

test1 = st.sidebar.number_input(
    "Test 1 (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=0.1,
    format="%.1f"
)

test2 = st.sidebar.number_input(
    "Test 2 (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=0.1,
    format="%.1f"
)

# ==================================================
# MODEL INPUT
# ==================================================

student = pd.DataFrame({
    # "Course Code": [course],
    "Test 1": [test1*0.20],
    "Test 2": [test2*0.20]
})

student_x = pd.get_dummies(
    student,
    drop_first=False
)

student_x = student_x.reindex(
    columns=feature_columns,
    fill_value=0
)


# ==================================================
# PREDICTION
# ==================================================

prob_pass = model.predict_proba(student_x)[0, 1]

# ==================================================
# MAIN LAYOUT
# ==================================================

left_col, right_col = st.columns([1, 1])

# ==================================================
# LEFT COLUMN
# ==================================================

with left_col:

    st.subheader("📈 Your Current Position")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Test 1", f"{test1:.1f}%")

    with c2:
        st.metric("Test 2", f"{test2:.1f}%")

    current_marks = (0.20 * test1 + 0.20 * test2)

    st.success(
        f"""
        You have currently secured **{current_marks:.1f}% out of the available 40%** before sitting the final examination.
        """
    )

    st.markdown("---")

    st.subheader("🎯 What Do I Need In The Final Exam?")

    planner = calculate_exam_requirements(test1, test2)

    rows = []

    for goal, exam_pct in planner.items():
        if exam_pct > 100:
            rows.append({
                "Desired Outcome": goal,
                "Exam Contribution Needed (out of 60%)": "Not Achievable"
            })
         
        else:
            marks_out_of_60 = round(exam_pct / 100 * 60, 1)
            rows.append({
                "Desired Outcome": goal,
                "Exam Contribution Needed (out of 60%)": f"{marks_out_of_60:.1f}%"
            })
    planner_df = pd.DataFrame(rows)

    st.dataframe(
        planner_df,
        hide_index=True,
        use_container_width=True
    )

    pass_pct = planner["Pass (≥ 50%)"]

    if pass_pct <= 100:

        pass_mark_60 = round(pass_pct / 100 * 60, 1)

        st.info(
            f"""
            To achieve a **Pass (≥ 50%)**, aim for approximately **{pass_mark_60:.1f}% in the final examination**.
            """
        )

# ==================================================
# RIGHT COLUMN
# ==================================================

with right_col:

    st.subheader("📊 Historical Passing Likelihood")

    st.caption(
        """
        This is an estimate based on previous students with similar Test 1 and Test 2 results and is not a guarantee of future performance.
        """
    )

    with st.expander("How reliable is this prediction?"):
        st.write(
            """
            This estimate is generated using a temporally validated, sigmoid-calibrated logistic regression model.

            During independent testing, the model correctly classified approximately **4 out of 5** students.

            Predictions are based on historical student data and provide estimates rather than guarantees of future performance.

            Students experiencing difficulties should seek academic support from unit coordinator, teaching staff, learning advisors, and other available support services.
            """
        )

    value = prob_pass * 100

    gauge = go.Figure()

    # Gauge dimensions
    outer_r = 1.0
    inner_r = 0.75

    # Colour bands: (start, end, colour)
    bands = [
        (0, 50, "#FF1744"),
        (50, 70, "#FF9100"),
        (70, 85, "#FFEA00"),
        (85, 100, "#00E676")
    ]

    # Draw semicircular coloured bands
    for start, end, colour in bands:

        theta = np.linspace(
            np.pi - np.pi * start / 100,
            np.pi - np.pi * end / 100,
            50
        )

        x = np.concatenate([
            outer_r * np.cos(theta),
            inner_r * np.cos(theta[::-1])
        ])

        y = np.concatenate([
            outer_r * np.sin(theta),
            inner_r * np.sin(theta[::-1])
        ])

        gauge.add_trace(go.Scatter(
            x=x,
            y=y,
            fill="toself",
            fillcolor=colour,
            line=dict(width=0),
            mode="lines",
            showlegend=False,
            hoverinfo="skip"
        ))

    # Calculate needle position
    angle = np.pi - np.pi * value / 100

    needle_length = 0.85

    needle_x = needle_length * np.cos(angle)
    needle_y = needle_length * np.sin(angle)

    # Draw needle
    gauge.add_trace(go.Scatter(
        x=[0, needle_x],
        y=[0, needle_y],
        mode="lines",
        line=dict(color="black", width=5),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Centre pivot
    gauge.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode="markers",
        marker=dict(size=20, color="black"),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Percentage display
    gauge.add_annotation(
        x=0,
        y=-0.22,
        text=f"<b>{value:.1f}%</b>",
        showarrow=False,
        font=dict(size=30, color="#1976D2")
    )

    # Scale labels
    for x, y, label in [
        (-1.02, -0.08, "0%"),
        (0, 1.07, "50%"),
        (1.02, -0.08, "100%")
    ]:
        gauge.add_annotation(
            x=x, y=y,
            text=label,
            showarrow=False,
            font=dict(size=13, color="#666666")
        )

    # Layout
    gauge.update_layout(
        # title="Likelihood of Passing",
        height=400,
        margin=dict(t=0, b=10, l=0, r=0),
        xaxis=dict(
            visible=False,
            range=[-1.15, 1.15],
            fixedrange=True
        ),
        yaxis=dict(
            visible=False,
            range=[-0.35, 1.15],
            scaleanchor="x",
            scaleratio=1,
            fixedrange=True
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    if prob_pass >= 0.85:

        st.success(
            """
            🟢 Very Strong Position

            Students with similar results have historically been very likely to pass. Keep up the good work!
            """
        )

    elif prob_pass >= 0.70:

        st.success(
            """
            🟢 Good Progress

            Students with similar results have generally performed well. Keep working towards your goal.
            """
        )

    elif prob_pass >= 0.50:

        st.warning(
            """
            🟠 Developing

            There is a reasonable chance of passing, but continued effort and preparation are important.
            """
        )

    else:

        st.error(
            """
            🔴 Needs Attention

            Students with similar results have historically faced greater difficulty passing. Focus on
            strengthening your understanding and preparing for the final examination.
            """
        )

# ==================================================
# ASSESSMENT STRUCTURE
# ==================================================

st.markdown("---")

st.subheader("🧮 Assessment Structure")

st.info(
    """
    **Final Mark** = 20% × Test 1 Mark + 20% × Test 2 Mark + 60% × Examination Mark
    """
)
