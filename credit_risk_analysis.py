import pandas as pd

df = pd.read_excel(
    "data/default of credit card clients.xls",
    header=1
)

print(df.shape)
print(df.head())

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDEFAULT DISTRIBUTION:")
print(df["default payment next month"].value_counts())

print("\nDEFAULT PROPORTIONS:")
print(df["default payment next month"].value_counts(normalize=True) * 100)

print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

print("\nDUPLICATE IDs:")
print(df["ID"].duplicated().sum())

print("\nSEX CATEGORIES:")
print(df["SEX"].value_counts().sort_index())

print("\nEDUCATION CATEGORIES:")
print(df["EDUCATION"].value_counts().sort_index())

print("\nMARRIAGE CATEGORIES:")
print(df["MARRIAGE"].value_counts().sort_index())

print("\nREPAYMENT STATUS: PAY_0")
print(df["PAY_0"].value_counts().sort_index())
print("\nTARGET COLUMN CHECK:")
target = "default payment next month"

print("Target column:", target)
print("Unique target values:", sorted(df[target].unique()))

print("\nEDUCATION AND MARRIAGE CODES TO REVIEW:")
print("Education codes:", sorted(df["EDUCATION"].unique()))
print("Marriage codes:", sorted(df["MARRIAGE"].unique()))


# Create a working copy of the dataset
df_clean = df.copy()

# Remove ID because it is only a customer identifier
df_clean = df_clean.drop(columns=["ID"])

# Group unusual education codes into category 4 (Other)
df_clean["EDUCATION"] = df_clean["EDUCATION"].replace({
    0: 4,
    5: 4,
    6: 4
})

# Group the unusual marriage code 0 into category 3 (Other)
df_clean["MARRIAGE"] = df_clean["MARRIAGE"].replace({
    0: 3
})

print("\nCLEANED DATASET:")
print("Shape:", df_clean.shape)

print("\nEducation categories after cleaning:")
print(df_clean["EDUCATION"].value_counts().sort_index())

print("\nMarriage categories after cleaning:")
print(df_clean["MARRIAGE"].value_counts().sort_index())

print("\nMissing values after cleaning:", df_clean.isnull().sum().sum())


# STEP 7: EXPLORE DEFAULT RATES BY REPAYMENT STATUS

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the default rate for each repayment status
default_by_repayment = df_clean.groupby("PAY_0")[
    "default payment next month"
].mean() * 100

print("\nDEFAULT RATE BY REPAYMENT STATUS:")
print(default_by_repayment.round(2))

# Plot the results
plt.figure(figsize=(9, 5))
sns.barplot(
    x=default_by_repayment.index,
    y=default_by_repayment.values
)

plt.title("Default Rate by Most Recent Repayment Status")
plt.xlabel("Repayment Status (PAY_0)")
plt.ylabel("Customers Who Defaulted (%)")
plt.tight_layout()
plt.show()


# STEP 8: DEFAULT RATE BY CREDIT LIMIT

# Divide customers into five groups based on credit limit
df_clean["credit_limit_group"] = pd.qcut(
    df_clean["LIMIT_BAL"],
    q=5,
    duplicates="drop"
)

# Calculate default rate for each group
default_by_limit = df_clean.groupby(
    "credit_limit_group",
    observed=True
)["default payment next month"].mean() * 100

print("\nDEFAULT RATE BY CREDIT LIMIT GROUP:")
print(default_by_limit.round(2))

# Plot the results
plt.figure(figsize=(10, 5))
default_by_limit.plot(kind="bar")

plt.title("Default Rate by Credit Limit Group")
plt.xlabel("Credit Limit Group")
plt.ylabel("Customers Who Defaulted (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# STEP 9: PREPARE FEATURES AND TARGET

from sklearn.model_selection import train_test_split

# Remove the temporary credit-limit group column.
# It was created only for our chart.
df_model = df_clean.drop(columns=["credit_limit_group"])

# X contains the information used to make predictions.
X = df_model.drop(columns=["default payment next month"])

# y contains the outcome we want to predict.
y = df_model["default payment next month"]

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nMACHINE LEARNING DATA PREPARATION:")
print("Feature data shape:", X.shape)
print("Target data shape:", y.shape)
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# STEP 10: TRAIN OUR FIRST MACHINE-LEARNING MODEL

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Scale the features using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the model
model = LogisticRegression(max_iter=1000)

# Train it using the training data
model.fit(X_train_scaled, y_train)

# Predict outcomes for the unseen test data
y_pred = model.predict(X_test_scaled)

# Evaluate the predictions
print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# STEP 11: TRAIN A DECISION TREE

from sklearn.tree import DecisionTreeClassifier

# Create the model
tree_model = DecisionTreeClassifier(
    max_depth=6,
    class_weight="balanced",
    random_state=42
)

# Train using the original, unscaled training features
tree_model.fit(X_train, y_train)

# Predict on the same unseen test set
tree_pred = tree_model.predict(X_test)

# Evaluate the results
print("\nDECISION TREE RESULTS")
print("Accuracy:", round(accuracy_score(y_test, tree_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, tree_pred))


# STEP 12: TRAIN A RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

# Create the model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the same test set
rf_pred = rf_model.predict(X_test)

# Evaluate the results
print("\nRANDOM FOREST RESULTS")
print("Accuracy:", round(accuracy_score(y_test, rf_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))


# STEP 13: TOP-10% CUSTOMER REVIEW EXPERIMENT

import pandas as pd

# Get each model's estimated probability of default (class 1)
lr_risk = model.predict_proba(X_test_scaled)[:, 1]
tree_risk = tree_model.predict_proba(X_test)[:, 1]
rf_risk = rf_model.predict_proba(X_test)[:, 1]

# Store actual outcomes and model risk scores together
risk_results = pd.DataFrame({
    "actual_default": y_test.to_numpy(),
    "Logistic Regression": lr_risk,
    "Decision Tree": tree_risk,
    "Random Forest": rf_risk
})

def evaluate_top_10_percent(data, risk_column):
    # Select the 10% of test customers with the highest risk scores
    review_count = int(len(data) * 0.10)

    selected = data.nlargest(review_count, risk_column)

    # Count actual defaulters among the selected customers
    defaulters_found = selected["actual_default"].sum()
    total_defaulters = data["actual_default"].sum()

    recall = defaulters_found / total_defaulters * 100
    precision = defaulters_found / review_count * 100

    return {
        "Customers reviewed": review_count,
        "Actual defaulters found": int(defaulters_found),
        "Defaulters captured (%)": round(recall, 2),
        "Default rate in reviewed group (%)": round(precision, 2)
    }

print("\nTOP-10% REVIEW CAPACITY RESULTS")

for model_name in [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest"
]:
    print(f"\n{model_name}:")
    print(evaluate_top_10_percent(risk_results, model_name))


# STEP 14: TEST DIFFERENT DECISION THRESHOLDS

from sklearn.metrics import precision_score, recall_score, f1_score

# Predicted probabilities for the default class
model_probabilities = {
    "Logistic Regression": lr_risk,
    "Decision Tree": tree_risk,
    "Random Forest": rf_risk
}

thresholds = [0.30, 0.40, 0.50]

print("\nTHRESHOLD ANALYSIS")

for model_name, probabilities in model_probabilities.items():
    print(f"\n{model_name}")

    for threshold in thresholds:
        # Convert probabilities into 0/1 predictions
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"F1: {f1:.3f}"
        )

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

comparison_rows = []

for model_name, probabilities in model_probabilities.items():
    predictions = (probabilities >= 0.50).astype(int)

    comparison_rows.append({
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, predictions), 3),
        "Precision": round(precision_score(y_test, predictions, zero_division=0), 3),
        "Recall": round(recall_score(y_test, predictions, zero_division=0), 3),
        "F1 Score": round(f1_score(y_test, predictions, zero_division=0), 3),
        "Top 10% Defaulters Found": evaluate_top_10_percent(
            risk_results, model_name
        )["Actual defaulters found"]
    })

comparison_df = pd.DataFrame(comparison_rows)

print("\nFINAL MODEL COMPARISON")
print(comparison_df.to_string(index=False))


# FEATURE IMPORTANCE: What information does the Random Forest use?

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 FEATURES USED BY RANDOM FOREST")
print(importance_df.head(10).to_string(index=False))

# Visualise the top 10 features
top_10 = importance_df.head(10).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 6))
plt.barh(top_10["Feature"], top_10["Importance"])
plt.xlabel("Feature Importance")
plt.title("Top 10 Features in Random Forest")
plt.tight_layout()
plt.show()
