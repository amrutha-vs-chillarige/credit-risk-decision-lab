from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# 1. PAGE CONFIGURATION AND THEME
# ============================================================

st.set_page_config(
    page_title="Credit Risk | Decision Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CREAM = "#F8F4EC"
BEIGE = "#E9DDCB"
SAND = "#C6A078"
BROWN = "#76543D"
DARK_BROWN = "#3F2B20"
MUTED_BROWN = "#8A7666"
CARD = "#FFFCF7"
BORDER = "#E3D6C5"

st.markdown(
    dedent(f"""
    <style>
    .stApp {{
        background-color: {CREAM};
        color: {DARK_BROWN};
    }}

    /* Match Streamlit's fixed top header to the site background. */
    header[data-testid="stHeader"] {{
        background: {CREAM} !important;
    }}

    [data-testid="stToolbar"] {{
        background: transparent !important;
    }}

    .block-container {{
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3 {{
        color: {DARK_BROWN} !important;
    }}

    p, li, label {{
        color: {DARK_BROWN};
    }}

    [data-testid="stMetric"] {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 15px;
        padding: 18px;
    }}

    [data-testid="stMetricLabel"] {{
        color: {MUTED_BROWN} !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {DARK_BROWN} !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {CARD};
        border-color: {BORDER};
        border-radius: 16px;
    }}

    .hero {{
        background: linear-gradient(115deg, #E9DDCB, #F4E9D9);
        border: 1px solid {BORDER};
        border-radius: 22px;
        padding: 30px;
        margin: 20px 0 26px 0;
    }}

    .eyebrow {{
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.78rem;
        font-weight: 700;
        color: {BROWN};
    }}

    .subtle {{
        color: {MUTED_BROWN};
        font-size: 0.94rem;
        line-height: 1.7;
    }}

    .info-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 15px;
        padding: 20px;
        height: 100%;
    }}

    .info-card h3 {{
        margin-top: 0;
        font-size: 1.12rem;
    }}

    .formula {{
        background: #F1E7D9;
        border-radius: 9px;
        padding: 12px 14px;
        font-family: monospace;
        overflow-wrap: anywhere;
        color: {DARK_BROWN};
        margin: 8px 0;
    }}

    div.stButton > button {{
        border-radius: 12px;
        border: 1px solid {BORDER};
        min-height: 44px;
    }}

    div.stButton > button[kind="primary"] {{
        background-color: {BROWN} !important;
        color: #FFFFFF !important;
        border-color: {BROWN} !important;
    }}

    div.stButton > button[kind="primary"] p,
    div.stButton > button[kind="primary"] span,
    div.stButton > button[kind="primary"] div {{
        color: #FFFFFF !important;
    }}

    div.stButton > button[kind="secondary"] {{
        background-color: {CARD} !important;
        color: {DARK_BROWN} !important;
        border-color: {BORDER} !important;
    }}

    div.stButton > button[kind="secondary"] p,
    div.stButton > button[kind="secondary"] span,
    div.stButton > button[kind="secondary"] div {{
        color: {DARK_BROWN} !important;
    }}

    hr {{
        border-color: {BORDER};
    }}

    [data-testid="stCaptionContainer"] {{
        color: {MUTED_BROWN};
    }}
    </style>
    """).strip(),
    unsafe_allow_html=True,
)

# ============================================================
# 2. HEADER
# ============================================================

logo_col, brand_col = st.columns([1, 9], vertical_alignment="center")

with logo_col:
    st.markdown(
        dedent("""
        <div style="
            background:#76543D;
            color:#FFFCF7;
            width:68px;
            height:68px;
            border-radius:18px;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:1.35rem;
            font-weight:800;
        ">CR</div>
        """).strip(),
        unsafe_allow_html=True,
    )

with brand_col:
    st.markdown(
        dedent("""
        <div style="font-weight:800;color:#3F2B20;font-size:1.35rem;">
            CREDIT RISK <span style="color:#76543D;">/ DECISION LAB</span>
        </div>
        <div style="font-size:0.9rem;color:#8A7666;margin-top:5px;">
            Machine learning · Evaluation · Decision support
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

st.write("")

# ============================================================
# 3. LOAD, CLEAN AND SPLIT DATA
# ============================================================

DATA_PATH = (
        Path(__file__).resolve().parent
        / "data"
        / "default of credit card clients.xls"
)


@st.cache_resource
def train_all_models(path_string):
    X, y, cleaned_df, target_column = load_dataset(path_string)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Standardisation is included in the logistic regression
    # pipeline because it is sensitive to feature scales.
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    probabilities = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.50).astype(int)

        probabilities[name] = prob
        predictions[name] = pred

        results[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, pred, zero_division=0
            ),
        }

    # The Random Forest is the primary model for the interactive
    # threshold, capacity and feature-importance analyses.
    rf_model = models["Random Forest"]
    rf_probabilities = probabilities["Random Forest"]

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test.reset_index(drop=True),
        "models": models,
        "results": results,
        "probabilities": probabilities,
        "predictions": predictions,
        "rf_model": rf_model,
        "rf_probabilities": rf_probabilities,
        "feature_names": X.columns.tolist(),
        "cleaned_df": cleaned_df,
        "target_column": target_column,
    }


@st.cache_data
def load_dataset(path_string):
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Check that the Excel file is inside the data folder."
        )

    # The first row in this dataset file contains a title.
    df = pd.read_excel(path, header=1)
    df.columns = df.columns.astype(str).str.strip()

    # Remove accidental whitespace from text column names.
    target_candidates = [
        col for col in df.columns
        if "default payment next month" in col.lower()
    ]

    if not target_candidates:
        raise ValueError(
            "Could not find the target column "
            "'default payment next month'. "
            "Please check the Excel column headings."
        )

    target_column = target_candidates[0]

    # Remove the ID because it is an identifier, not a useful
    # customer-risk measurement.
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Handle undocumented / grouped categories consistently.
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace(
            {0: 4, 5: 4, 6: 4}
        )

    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # Put the target at the end and ensure it is binary.
    y = df[target_column].astype(int)
    X = df.drop(columns=[target_column])

    if y.nunique() != 2:
        raise ValueError(
            "Expected a binary target with values 0 and 1."
        )

    return X, y, df, target_column


try:
    with st.spinner("Preparing the dataset and training models..."):
        data = train_all_models(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not prepare the project: {exc}")
    st.info(
        "Check the dataset path and make sure the required "
        "Python packages are installed."
    )
    st.stop()

X = data["X"]
y = data["y"]
y_test = data["y_test"]
rf_probabilities = data["rf_probabilities"]
rf_model = data["rf_model"]

n_customers = len(y)
overall_default_rate = float(y.mean())

# ============================================================
# 4. NAVIGATION
# ============================================================

pages = [
    "Overview",
    "Models",
    "Thresholds",
    "Capacity",
    "Features",
    "Findings",
]

if "main_navigation" not in st.session_state:
    st.session_state["main_navigation"] = "Overview"

nav_cols = st.columns(len(pages))

for i, page in enumerate(pages):
    with nav_cols[i]:
        is_active = st.session_state["main_navigation"] == page

        if st.button(
                page,
                key=f"nav_{page}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
        ):
            st.session_state["main_navigation"] = page
            st.rerun()

st.divider()

page = st.session_state["main_navigation"]


# ============================================================
# 5. REUSABLE EXPLANATION COMPONENTS
# ============================================================

def section_intro(title, description):
    st.title(title)
    st.markdown(
        f'<p class="subtle">{description}</p>',
        unsafe_allow_html=True,
    )


def explanation_card(title, body):
    # dedent prevents Markdown from treating indented HTML as a code block.
    body_html = dedent(body).strip()
    html = dedent(f"""
    <div class="info-card">
        <h3>{title}</h3>
        <div style="line-height:1.7;">{body_html}</div>
    </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)


def formula(text):
    st.markdown(
        f'<div class="formula">{text}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        dedent("""
        <div class="hero">
            <div class="eyebrow">Credit risk analytics project</div>
            <h1 style="margin-top:10px;">
                Turning credit data into better-informed decisions.
            </h1>
            <p style="font-size:1.05rem;line-height:1.8;">
                This project uses historical customer data to estimate
                the risk of credit-card payment default. It compares
                machine-learning models, examines classification
                errors and explores how a bank could prioritise a
                limited number of customer reviews.
            </p>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Customers in dataset", f"{n_customers:,}")
    c2.metric(
        "Historical default rate",
        f"{overall_default_rate:.1%}",
    )
    c3.metric("Input features", f"{X.shape[1]}")

    st.caption(
        "The default rate is the share of records labelled as "
        "default=1 in the full dataset. It is not a prediction "
        "of future population-wide default."
    )

    st.subheader("What does this project do?")

    col1, col2 = st.columns(2)

    with col1:
        explanation_card(
            "01 · Predict risk",
            """
            Train three classification models to estimate the
            probability that a customer will default on a payment.
            The model uses customer information, repayment history,
            bill amounts and previous payment amounts.
            """,
        )

    with col2:
        explanation_card(
            "02 · Support decisions",
            """
            Explore how the probability threshold and the number of
            customers reviewed change the balance between finding
            defaulters and flagging customers who would not default.
            """,
        )

    st.write("")

    st.subheader("How to explore the project")

    overview_cols = st.columns(3)

    with overview_cols[0]:
        st.markdown("**Model comparison**")
        st.write(
            "Compare Logistic Regression, Decision Tree and "
            "Random Forest using accuracy, precision, recall and F1."
        )
        if st.button("Explore models →", use_container_width=True):
            st.session_state["main_navigation"] = "Models"
            st.rerun()

    with overview_cols[1]:
        st.markdown("**Threshold analysis**")
        st.write(
            "Change the probability cutoff and see how precision, "
            "recall and the number flagged respond."
        )
        if st.button("Explore thresholds →", use_container_width=True):
            st.session_state["main_navigation"] = "Thresholds"
            st.rerun()

    with overview_cols[2]:
        st.markdown("**Review capacity**")
        st.write(
            "Simulate reviewing only the highest-risk 5–30% of "
            "customers and measure how many actual defaulters "
            "are found."
        )
        if st.button("Explore capacity →", use_container_width=True):
            st.session_state["main_navigation"] = "Capacity"
            st.rerun()

    st.subheader("Dataset and methodology")

    st.markdown(
        """
        - **Dataset:** UCI Default of Credit Card Clients.
        - **Context:** credit-card customers in Taiwan; historical
          data covering April–September 2005.
        - **Target:** `default payment next month` — 1 means a
          recorded default, and 0 means no recorded default.
        - **Split:** 80% training data and 20% test data, with
          stratification to preserve the target-class proportion.
        - **Random seed:** 42, for reproducibility.
        - **Preprocessing:** remove the ID field and group certain
          undocumented education and marriage codes into an
          "other/unknown" category.
        """
    )

    st.markdown(
        "[Open the official UCI dataset page](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients)"
    )

    st.warning(
        "This is historical data, not a live banking dataset. "
        "The models are educational decision-support experiments, "
        "not validated systems for making real lending decisions."
    )


# ============================================================
# PAGE 2: MODEL COMPARISON
# ============================================================

elif page == "Models":

    section_intro(
        "Model Performance Comparison",
        "Three classifiers are evaluated on the same held-out test "
        "set. All displayed scores are calculated from this run. "
        "The classification threshold is 0.50 for this comparison.",
    )

    st.subheader("The three models")

    model_cols = st.columns(3)

    with model_cols[0]:
        explanation_card(
            "Logistic Regression",
            """
            <b>How it works:</b> estimates a relationship between
            input features and the probability of default using a
            logistic function.

            <br><br><b>Strength:</b> relatively easy to interpret
            and a useful baseline.

            <br><br><b>Limitation:</b> its standard form models
            relatively simple relationships unless extra features
            or transformations are added.
            """,
        )

    with model_cols[1]:
        explanation_card(
            "Decision Tree",
            """
            <b>How it works:</b> repeatedly splits customers into
            groups using feature-based rules. Each path through the
            tree leads to a predicted class or probability.

            <br><br><b>Strength:</b> captures nonlinear patterns
            and can be explained as a sequence of rules.

            <br><br><b>Limitation:</b> a single tree can overfit
            training data and may be unstable.
            """,
        )

    with model_cols[2]:
        explanation_card(
            "Random Forest",
            """
            <b>How it works:</b> trains many decision trees on
            varied samples and feature subsets, then combines their
            predictions.

            <br><br><b>Strength:</b> can capture complex patterns
            and reduce the instability of a single tree.

            <br><br><b>Limitation:</b> is harder to interpret as
            one simple set of rules.
            """,
        )

    st.subheader("What do the performance metrics mean?")

    with st.expander("Accuracy — overall correctness", expanded=True):
        st.markdown(
            """
            **Question answered:** Out of all test customers, what
            proportion did the model classify correctly?

            It counts both actual defaulters correctly flagged and
            non-defaulters correctly left unflagged.

            """
        )
        formula("Accuracy = (TP + TN) / (TP + TN + FP + FN)")
        st.markdown(
            """
            - **TP:** actual defaulters predicted as defaulters.
            - **TN:** actual non-defaulters predicted as non-defaulters.
            - **FP:** non-defaulters incorrectly flagged as defaulters.
            - **FN:** actual defaulters the model missed.

            **Interpretation:** accuracy is easy to understand, but
            it can hide poor performance on the less common class.
            """
        )

    with st.expander("Precision — how reliable are the flags?"):
        st.markdown(
            """
            **Question answered:** Of the customers flagged as
            likely to default, what proportion actually defaulted?
            """
        )
        formula("Precision = TP / (TP + FP)")
        st.markdown(
            """
            **Example:** if 100 customers are flagged and 60 of
            them actually default, precision is 60%.

            Higher precision means fewer false alarms among the
            flagged group. It matters when investigating a flagged
            customer is costly or time-consuming.
            """
        )

    with st.expander("Recall — how many defaulters did we find?"):
        st.markdown(
            """
            **Question answered:** Of all customers who actually
            defaulted, what proportion did the model identify?
            """
        )
        formula("Recall = TP / (TP + FN)")
        st.markdown(
            """
            **Example:** if 100 customers actually default and the
            model flags 70 of them, recall is 70%.

            Higher recall means fewer actual defaulters are missed.
            It can be valuable when missed defaults are costly.
            """
        )

    with st.expander("F1-score — the balance between precision and recall"):
        st.markdown(
            """
            F1 combines precision and recall into one score using
            their harmonic mean. It is useful when we want to
            balance false alarms against missed defaulters.
            """
        )
        formula(
            "F1 = 2 × (Precision × Recall) / "
            "(Precision + Recall)"
        )
        st.markdown(
            """
            **Example:** precision = 0.60 and recall = 0.30 gives:

            F1 = 2 × (0.60 × 0.30) / (0.60 + 0.30) = 0.40.

            F1 is high only when both precision and recall are
            reasonably strong. It does not include true negatives,
            and it does not account for the different financial
            costs of each error.
            """
        )

    st.info(
        "TP, TN, FP and FN are counted on the held-out test set. "
        "The positive class is default=1. The standard threshold "
        "of 0.50 is used for this model comparison."
    )

    st.subheader("Performance results")

    rows = []
    for model_name, metrics in data["results"].items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1 Score": metrics["F1 Score"],
            }
        )

    results_df = pd.DataFrame(rows)

    display_df = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_df[col] = display_df[col].map(
            lambda value: f"{value:.1%}"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These results are calculated dynamically. Model settings, "
        "data cleaning, the train/test split and random seed affect "
        "the results, so figures may differ from earlier runs."
    )

    st.subheader("All model metrics at a glance")

    st.markdown(
        """
        Each group compares the three models on one metric.
        Hover over a bar to see its exact value.

        - **Accuracy:** overall proportion classified correctly.
        - **Precision:** proportion of flagged customers who actually defaulted.
        - **Recall:** proportion of actual defaulters identified.
        - **F1-score:** balance between precision and recall.
        """
    )

    chart_df = results_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        chart_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text=chart_df["Score"].map(lambda v: f"{v:.1%}"),
        color_discrete_map={
            "Logistic Regression": BROWN,
            "Decision Tree": SAND,
            "Random Forest": "#A88D78",
        },
        hover_data={"Score": ":.4f", "Model": True, "Metric": True},
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Metric: %{x}<br>"
            "Score: %{y:.2%}"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        xaxis_title="Evaluation metric",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.12], tickformat=".0%"),
        plot_bgcolor=CARD,
        paper_bgcolor=CREAM,
        font_color=DARK_BROWN,
        legend_title_text="Model",
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=75, b=55),
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed model comparison")

    display_df = results_df.copy()

    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_df[col] = display_df[col].map(
            lambda value: f"{value:.1%}"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "All values are calculated from the same held-out test set "
        "using a 0.50 probability threshold. Hover over a bar for "
        "the exact underlying score."
    )

    st.subheader("Confusion matrices: all three models")

    st.markdown(
        """
        Each matrix compares actual outcomes with model predictions.

        - **True negative (top-left):** correctly identified non-defaulter.
        - **False positive (top-right):** non-defaulter incorrectly flagged.
        - **False negative (bottom-left):** actual defaulter missed.
        - **True positive (bottom-right):** correctly identified defaulter.
        """
    )

    matrix_cols = st.columns(3)

    for col, (model_name, predictions) in zip(
            matrix_cols, data["predictions"].items()
    ):
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No default", "Actual: Default"],
            columns=["Predicted: No default", "Predicted: Default"],
        )

        with col:
            st.markdown(f"**{model_name}**")
            st.dataframe(cm_df, use_container_width=True)

    st.subheader("How should we choose a model?")

    st.markdown(
        """
        There is no universally best metric. If missed defaulters are
        the primary concern, recall may matter more. If investigation
        resources are expensive, precision may matter more. F1 provides
        a useful balance, but real deployment would also require
        financial cost analysis, calibration, fairness testing and
        validation on newer data.
        """
    )


# ============================================================
# PAGE 3: THRESHOLD ANALYSIS
# ============================================================

elif page == "Thresholds":

    section_intro(
        "Decision Threshold Analysis",
        "The Random Forest estimates a probability for each customer. "
        "The threshold converts that probability into a high-risk "
        "or not-high-risk classification.",
    )

    st.markdown(
        """
        A threshold is a decision rule, not a change to the trained
        model itself.

        - If predicted probability is **at least the threshold**,
          the customer is flagged.
        - Otherwise, the customer is not flagged.

        Lower thresholds usually identify more actual defaulters but
        also flag more non-defaulters. Higher thresholds usually
        reduce false alarms but miss more actual defaulters.
        """
    )

    threshold = st.slider(
        "Choose the probability threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        format="%.2f",
        key="risk_threshold",
    )

    predictions_at_threshold = (
            rf_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions_at_threshold,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        predictions_at_threshold,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        predictions_at_threshold,
        zero_division=0,
    )
    accuracy = accuracy_score(
        y_test,
        predictions_at_threshold,
    )

    flagged_count = int(predictions_at_threshold.sum())
    test_count = len(y_test)
    flagged_share = flagged_count / test_count

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{precision:.1%}")
    c2.metric("Recall", f"{recall:.1%}")
    c3.metric("F1-score", f"{f1:.3f}")
    c4.metric("Customers flagged", f"{flagged_count:,}")

    st.caption(
        f"Threshold = {threshold:.2f}. The model flags "
        f"{flagged_share:.1%} of the test customers. "
        "The metrics above are recalculated from the test labels "
        "and the selected threshold."
    )

    with st.expander("What do these four values mean?"):
        st.markdown(
            f"""
            - **Precision ({precision:.1%}):** of the customers
              flagged at this threshold, this share actually defaulted.
            - **Recall ({recall:.1%}):** of all actual defaulters
              in the test set, this share was flagged.
            - **F1-score ({f1:.3f}):** the harmonic mean of
              precision and recall.
            - **Customers flagged ({flagged_count:,}):** the number
              of test customers whose predicted probability is
              at least {threshold:.2f}.
            """
        )

    st.subheader("Precision and recall across thresholds")

    threshold_values = np.round(
        np.arange(0.10, 0.901, 0.05), 2
    )

    curve_rows = []

    for t in threshold_values:
        pred = (rf_probabilities >= t).astype(int)
        curve_rows.append(
            {
                "Threshold": float(t),
                "Precision": precision_score(
                    y_test, pred, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, pred, zero_division=0
                ),
                "F1-score": f1_score(
                    y_test, pred, zero_division=0
                ),
                "Customers flagged": int(pred.sum()),
            }
        )

    curve_df = pd.DataFrame(curve_rows)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=curve_df["Threshold"],
            y=curve_df["Precision"],
            mode="lines+markers",
            name="Precision",
            line=dict(color=BROWN, width=3),
            marker=dict(size=8),
            customdata=np.column_stack(
                [
                    curve_df["Recall"],
                    curve_df["F1-score"],
                    curve_df["Customers flagged"],
                ]
            ),
            hovertemplate=(
                "<b>Threshold: %{x:.2f}</b><br>"
                "Precision: %{y:.1%}<br>"
                "Recall: %{customdata[0]:.1%}<br>"
                "F1-score: %{customdata[1]:.3f}<br>"
                "Customers flagged: %{customdata[2]:.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=curve_df["Threshold"],
            y=curve_df["Recall"],
            mode="lines+markers",
            name="Recall",
            line=dict(color=SAND, width=3),
            marker=dict(size=8),
            customdata=np.column_stack(
                [
                    curve_df["Precision"],
                    curve_df["F1-score"],
                    curve_df["Customers flagged"],
                ]
            ),
            hovertemplate=(
                "<b>Threshold: %{x:.2f}</b><br>"
                "Recall: %{y:.1%}<br>"
                "Precision: %{customdata[0]:.1%}<br>"
                "F1-score: %{customdata[1]:.3f}<br>"
                "Customers flagged: %{customdata[2]:.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=MUTED_BROWN,
        annotation_text=f"Selected: {threshold:.2f}",
        annotation_position="top",
    )

    fig.update_layout(
        xaxis_title="Probability threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.04], tickformat=".0%"),
        xaxis=dict(tickformat=".2f"),
        hovermode="x",
        plot_bgcolor=CARD,
        paper_bgcolor=CREAM,
        font_color=DARK_BROWN,
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=70, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "Hover over a point to see its exact threshold, precision, "
        "recall, F1-score and number of flagged customers. "
        "The dashed line marks your selected threshold."
    )

    st.subheader("Full threshold table")

    table_df = curve_df.copy()
    table_df["Threshold"] = table_df["Threshold"].map(
        lambda value: f"{value:.2f}"
    )
    for col in ["Precision", "Recall"]:
        table_df[col] = table_df[col].map(
            lambda value: f"{value:.1%}"
        )
    table_df["F1-score"] = table_df["F1-score"].map(
        lambda value: f"{value:.3f}"
    )
    table_df["Customers flagged"] = table_df[
        "Customers flagged"
    ].map(lambda value: f"{value:,}")

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.subheader("How should a threshold be chosen?")

    st.markdown(
        """
        A lower threshold may be suitable when missing a defaulter is
        considered especially costly. A higher threshold may be useful
        when investigation capacity is limited or false alarms are
        expensive. The appropriate value depends on operational
        capacity, error costs and the reliability of the probabilities.

        **Important:** a flagged customer is not necessarily going
        to default. This model estimates risk; it does not establish
        certainty or justify an automatic rejection.
        """
    )


# ============================================================
# PAGE 4: REVIEW CAPACITY
# ============================================================

elif page == "Capacity":

    section_intro(
        "Review Capacity and Risk Prioritisation",
        "What if a bank can only review a fraction of its customers? "
        "This analysis ranks test customers by predicted default "
        "probability and examines the highest-risk group.",
    )

    st.markdown(
        """
        Instead of flagging everyone above a fixed threshold, this
        scenario sorts customers from highest to lowest predicted
        risk and selects the top percentage for review.

        For example, a 10% review capacity means that only the
        highest-risk 10% of the test set are selected. This is a
        prioritisation exercise, not a recommendation to reject
        those customers.
        """
    )

    capacity = st.slider(
        "Percentage of customers available for review",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        format="%d%%",
        key="review_capacity",
    )

    review_count = max(
        1,
        int(np.ceil(len(y_test) * capacity / 100)),
    )

    ranked_indices = np.argsort(rf_probabilities)[::-1]
    selected_indices = ranked_indices[:review_count]

    selected_actual = y_test.iloc[selected_indices].to_numpy()
    selected_probabilities = rf_probabilities[selected_indices]

    actual_defaulters_in_test = int(y_test.sum())
    defaulters_found = int(selected_actual.sum())

    default_rate_selected = float(selected_actual.mean())
    capture_rate = (
        defaulters_found / actual_defaulters_in_test
        if actual_defaulters_in_test > 0
        else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers reviewed", f"{review_count:,}")
    c2.metric("Actual defaulters found", f"{defaulters_found:,}")
    c3.metric("Defaulters captured", f"{capture_rate:.1%}")
    c4.metric("Default rate in selected group", f"{default_rate_selected:.1%}")

    with st.expander("How are these metrics calculated?", expanded=True):
        st.markdown(
            f"""
            **Customers reviewed**

            Number of test customers selected for review.

            """
        )
        formula(
            "Review count = ceiling(test customers × capacity %)"
        )

        st.markdown(
            f"""
            **Actual defaulters found ({defaulters_found:,})**

            Count of customers in the selected group whose recorded
            target is default=1.

            **Defaulters captured ({capture_rate:.1%})**

            Percentage of all actual defaulters in the test set who
            are present in the selected group.
            """
        )
        formula(
            "Capture rate = defaulters found / all actual defaulters"
        )

        st.markdown(
            f"""
            **Default rate in the selected group ({default_rate_selected:.1%})**

            Percentage of selected customers who actually defaulted.
            This is the precision of this top-risk group, using the
            actual labels for evaluation.
            """
        )
        formula(
            "Selected-group default rate = defaulters found / customers reviewed"
        )

    st.subheader("Capacity versus outcomes")

    capacity_values = list(range(5, 31, 5))
    capacity_rows = []

    for cap in capacity_values:
        count = max(
            1,
            int(np.ceil(len(y_test) * cap / 100)),
        )
        indices = ranked_indices[:count]
        labels = y_test.iloc[indices].to_numpy()
        found = int(labels.sum())

        capacity_rows.append(
            {
                "Review capacity (%)": cap,
                "Customers reviewed": count,
                "Actual defaulters found": found,
                "Capture rate": (
                    found / actual_defaulters_in_test
                    if actual_defaulters_in_test
                    else 0
                ),
                "Default rate in selected group": float(labels.mean()),
            }
        )

    capacity_df = pd.DataFrame(capacity_rows)

    fig_capacity = go.Figure()

    fig_capacity.add_trace(
        go.Scatter(
            x=capacity_df["Review capacity (%)"],
            y=capacity_df["Capture rate"],
            mode="lines+markers",
            name="Defaulters captured",
            line=dict(color=BROWN, width=3),
            customdata=np.column_stack(
                [
                    capacity_df["Customers reviewed"],
                    capacity_df["Actual defaulters found"],
                    capacity_df["Default rate in selected group"],
                ]
            ),
            hovertemplate=(
                "<b>Review capacity: %{x}%</b><br>"
                "Capture rate: %{y:.1%}<br>"
                "Customers reviewed: %{customdata[0]:.0f}<br>"
                "Actual defaulters found: %{customdata[1]:.0f}<br>"
                "Selected-group default rate: %{customdata[2]:.1%}"
                "<extra></extra>"
            ),
        )
    )

    fig_capacity.add_trace(
        go.Scatter(
            x=capacity_df["Review capacity (%)"],
            y=capacity_df["Default rate in selected group"],
            mode="lines+markers",
            name="Default rate in selected group",
            line=dict(color=SAND, width=3),
            hovertemplate=(
                "Review capacity: %{x}%<br>"
                "Selected-group default rate: %{y:.1%}"
                "<extra></extra>"
            ),
        )
    )

    fig_capacity.update_layout(
        xaxis_title="Review capacity (%)",
        yaxis_title="Rate",
        yaxis=dict(tickformat=".0%", range=[0, 1.04]),
        hovermode="x",
        plot_bgcolor=CARD,
        paper_bgcolor=CREAM,
        font_color=DARK_BROWN,
        legend=dict(orientation="h", y=1.12),
    )

    st.plotly_chart(fig_capacity, use_container_width=True)

    st.caption(
        "Hover over a point for the exact values. Increasing review "
        "capacity generally finds more defaulters, but the additional "
        "customers selected tend to have lower predicted risk."
    )

    st.subheader("Capacity scenario table")

    capacity_display = capacity_df.copy()
    capacity_display["Capture rate"] = capacity_display[
        "Capture rate"
    ].map(lambda value: f"{value:.1%}")
    capacity_display["Default rate in selected group"] = (
        capacity_display["Default rate in selected group"].map(
            lambda value: f"{value:.1%}"
        )
    )

    st.dataframe(
        capacity_display,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "The selected group is evaluated using known outcomes from "
        "the test set. In a real future deployment, those outcomes "
        "would not be known at the time of selection."
    )


# ============================================================
# PAGE 5: FEATURE IMPORTANCE
# ============================================================

elif page == "Features":

    section_intro(
        "What Drives the Model's Predictions?",
        "Feature importance ranks the input variables the trained "
        "Random Forest relied on most when creating its decision trees.",
    )

    st.markdown(
        """
        Each feature describes an aspect of a customer or their
        credit-card history. The Random Forest uses these variables
        to split customers into groups with different patterns of
        the target outcome.
        """
    )

    st.subheader("What does feature importance mean?")

    explanation_card(
        "The key idea",
        """
        A higher importance value means the feature contributed
        more to the model's total reduction in classification
        impurity across the forest's trees.

        A lower value means the feature contributed less by this
        particular importance measure. It does not necessarily
        mean the feature is useless.
        """,
    )

    st.write("")

    st.markdown("### How is the number calculated?")

    st.markdown(
        """
        This chart uses the Random Forest's built-in
        **mean decrease in impurity** importance.

        A decision tree repeatedly splits records into groups.
        At each split, it seeks a division that makes the resulting
        groups more concentrated in one target class.
        """
    )

    formula(
        "Weighted impurity decrease = "
        "weighted impurity before split − weighted impurity after split"
    )

    st.markdown(
        """
        The algorithm accumulates the weighted impurity reductions
        associated with each feature across the trees, then normalises
        the values so the importances sum to approximately 1.

        Therefore:
        """
    )

    formula(
        "Normalised importance = feature's accumulated importance "
        "/ total accumulated importance"
    )

    st.markdown(
        """
        **Example:** an importance of 0.20 means the feature accounts
        for about 20% of the total normalised impurity-importance
        score across the fitted forest. It does **not** mean that
        the feature causes 20% of defaults or raises default risk
        by 20%.
        """
    )

    importance_df = pd.DataFrame(
        {
            "Feature": data["feature_names"],
            "Importance": rf_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    st.subheader("Interactive feature importance chart")

    top_n = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=min(23, len(importance_df)),
        value=min(10, len(importance_df)),
        step=1,
    )

    top_features = importance_df.head(top_n).sort_values(
        "Importance", ascending=True
    )

    fig_features = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#E9DDCB", "#C6A078", "#76543D"],
        text=top_features["Importance"].map(
            lambda value: f"{value:.4f}"
        ),
        hover_data={"Importance": ":.5f", "Feature": True},
        title=f"Top {top_n} Random Forest features",
    )

    fig_features.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Normalised importance: %{x:.5f}<br>"
            "Approximate share: %{x:.2%}"
            "<extra></extra>"
        ),
    )

    fig_features.update_layout(
        xaxis_title="Normalised impurity importance",
        yaxis_title="Feature",
        plot_bgcolor=CARD,
        paper_bgcolor=CREAM,
        font_color=DARK_BROWN,
        coloraxis_showscale=False,
        margin=dict(l=30, r=60, t=60, b=50),
    )

    st.plotly_chart(fig_features, use_container_width=True)

    st.success(
        "Hover over any bar to see the feature's exact importance "
        "value and its share of the total normalised importance."
    )

    st.subheader("Feature importance table")

    importance_display = importance_df.copy()
    importance_display["Importance"] = (
        importance_display["Importance"].map(
            lambda value: f"{value:.5f}"
        )
    )

    st.dataframe(
        importance_display,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Feature dictionary")

    st.markdown(
        "Every input variable is listed below with its meaning and "
        "its calculated importance in the trained Random Forest."
    )

    feature_definitions = {
        "LIMIT_BAL": "Credit limit granted to the customer.",
        "SEX": "Coded demographic field in the original dataset.",
        "EDUCATION": "Coded education category.",
        "MARRIAGE": "Coded marital-status category.",
        "AGE": "Customer age in years.",
        "PAY_0": "Repayment status in September 2005 (the original dataset uses PAY_0 for this month).",
        "PAY_2": "Repayment status in August 2005.",
        "PAY_3": "Repayment status in July 2005.",
        "PAY_4": "Repayment status in June 2005.",
        "PAY_5": "Repayment status in May 2005.",
        "PAY_6": "Repayment status in April 2005.",
        "BILL_AMT1": "Bill statement amount for September 2005.",
        "BILL_AMT2": "Bill statement amount for August 2005.",
        "BILL_AMT3": "Bill statement amount for July 2005.",
        "BILL_AMT4": "Bill statement amount for June 2005.",
        "BILL_AMT5": "Bill statement amount for May 2005.",
        "BILL_AMT6": "Bill statement amount for April 2005.",
        "PAY_AMT1": "Previous payment amount in September 2005.",
        "PAY_AMT2": "Previous payment amount in August 2005.",
        "PAY_AMT3": "Previous payment amount in July 2005.",
        "PAY_AMT4": "Previous payment amount in June 2005.",
        "PAY_AMT5": "Previous payment amount in May 2005.",
        "PAY_AMT6": "Previous payment amount in April 2005.",
    }

    feature_table = importance_df.copy()
    feature_table["Meaning"] = feature_table["Feature"].map(
        feature_definitions
    ).fillna("Input variable used by the model.")
    feature_table["Importance (%)"] = (
            feature_table["Importance"] * 100
    ).map(lambda value: f"{value:.2f}%")
    feature_table["Importance"] = feature_table["Importance"].map(
        lambda value: f"{value:.5f}"
    )
    feature_table = feature_table[
        ["Feature", "Meaning", "Importance", "Importance (%)"]
    ]

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("How should these values be interpreted?")

    st.markdown(
        """
        - **Higher value:** the feature contributed more to the
          forest's accumulated impurity reductions.
        - **Lower value:** the feature contributed less by this
          specific calculation.
        - **Zero or close to zero:** the feature was not used much,
          or contributed very little, under this model configuration.
        - **Similar features:** correlated variables may share or
          redistribute importance.
        """
    )

    st.warning(
        "Feature importance is not causation, not the direction of "
        "an effect, and not an individual customer's explanation. "
        "Impurity-based importance can favour features with many "
        "possible split points. A stronger follow-up analysis could "
        "compare it with permutation importance on held-out data."
    )


# ============================================================
# PAGE 6: FINDINGS AND LIMITATIONS
# ============================================================

elif page == "Findings":

    section_intro(
        "Findings, Interpretation and Limitations",
        "A summary of the project's current analytical approach, "
        "what its outputs can tell us and what they cannot establish.",
    )

    st.subheader("What this project investigates")

    st.markdown(
        """
        1. Whether different classification models can identify
           customers who subsequently defaulted in the historical data.
        2. How precision and recall change when the decision threshold
           changes.
        3. How much of the test set's actual default population could
           be found when reviewing only the highest-risk customers.
        4. Which features the Random Forest used most in its tree splits.
        """
    )

    st.subheader("How to interpret the results")

    explanation_card(
        "Performance is a trade-off",
        """
        Accuracy alone is not enough to assess a credit-risk model.
        Precision measures how reliable the flagged group is, while
        recall measures how many actual defaulters are found.
        F1 summarises the balance between precision and recall.
        """,
    )

    st.write("")

    explanation_card(
        "A threshold is a policy choice",
        """
        Lower thresholds tend to flag more customers and find more
        defaulters, but can increase false alarms. Higher thresholds
        tend to reduce flags while missing more actual defaulters.
        The best threshold depends on costs, resources and policy.
        """,
    )

    st.write("")

    explanation_card(
        "Capacity changes the outcome",
        """
        Reviewing only the highest-risk customers can concentrate
        recorded defaults within a smaller group. However, some
        actual defaulters will remain outside that group.
        The capacity chart quantifies this trade-off on test data.
        """,
    )

    st.write("")

    explanation_card(
        "Feature importance describes model behaviour",
        """
        The Random Forest's importance values show which features
        contributed most to its impurity reductions. They do not prove
        that a variable causes default, and they are not a substitute
        for a causal analysis.
        """,
    )

    st.subheader("Limitations")

    st.markdown(
        """
        - **Historical data:** the dataset relates to Taiwan in
          2005. It does not represent current customers or today's
          lending environment.
        - **Generalisation:** results on a random held-out sample
          may not reflect performance on future customers.
        - **Threshold assumptions:** the selected threshold is a
          modelling choice; it has not been optimised against a
          validated business cost function.
        - **Class imbalance:** default is the minority class, so
          accuracy alone can be misleading.
        - **Fairness:** demographic fields are present in the source
          data. This project does not establish that the model is
          fair or appropriate for actual lending decisions.
        - **Interpretability:** impurity-based feature importance
          can be affected by correlated predictors and features with
          many possible split points.
        - **No causal claims:** predictive associations do not
          establish that changing a feature would change default risk.
        """
    )

    st.subheader("What would improve the project further?")

    st.markdown(
        """
        - Validate on a later time period rather than relying only
          on a random train/test split.
        - Compare against a simple majority-class baseline.
        - Calibrate predicted probabilities and evaluate calibration.
        - Use permutation importance or SHAP for additional
          feature-level explanations.
        - Assess fairness and error rates across relevant groups.
        - Define financial costs for false positives and false
          negatives before choosing an operational threshold.
        - Test the system on current, representative data before
          considering real-world use.
        """
    )

    st.subheader("Dataset source")

    st.markdown(
        """
        The dataset is the **Default of Credit Card Clients**
        collection from the UCI Machine Learning Repository.
        The source documents 30,000 instances and a historical
        credit-card default prediction task.

        [View the official UCI dataset and metadata](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients)
        """
    )

    st.info(
        "This portfolio project demonstrates machine-learning "
        "evaluation and decision analysis. It is not a validated "
        "credit-scoring product and should not be used on its own "
        "to approve, reject or price real loans."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    dedent(f"""
    <div style="
        text-align:center;
        color:{MUTED_BROWN};
        font-size:0.85rem;
        line-height:1.8;
    ">
        <b style="color:{BROWN};">CREDIT RISK / DECISION LAB</b><br>
        Machine learning · Model evaluation · Decision support<br>
        UCI Default of Credit Card Clients · Historical data, 2005
    </div>
    """).strip(),
    unsafe_allow_html=True,
)
