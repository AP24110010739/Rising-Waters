"""
Rising Waters: Machine Learning Flood Prediction Model
=======================================================
This script covers:
  Epic 2: Data Visualization & Analysis
  Epic 3: Data Pre-processing
  Epic 4: Model Building & Selection
  Saves: floods.save (XGBoost model) and scaler.save
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# 0. PATHS
# ─────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "dataset", "rainfall_india.csv")
MODEL  = os.path.join(BASE, "model",   "floods.save")
SCALER = os.path.join(BASE, "model",   "scaler.save")
VIZ    = os.path.join(BASE, "static",  "images")
os.makedirs(VIZ, exist_ok=True)
os.makedirs(os.path.join(BASE, "model"), exist_ok=True)

print("=" * 60)
print("  RISING WATERS — FLOOD PREDICTION ML PIPELINE")
print("=" * 60)

# ─────────────────────────────────────────────
# EPIC 2.1 – Load & Explore Dataset
# ─────────────────────────────────────────────
print("\n[EPIC 2] Data Exploration & Visualization")
df = pd.read_csv(DATA)

# Normalise target column: CSV uses 'flood', project uses 'Floods'
if 'flood' in df.columns and 'Floods' not in df.columns:
    df.rename(columns={'flood': 'Floods'}, inplace=True)

print(f"\nDataset Shape : {df.shape}")
print(f"\nFirst 5 Rows:\n{df.head()}")
print(f"\nData Info:")
df.info()
print(f"\nDescriptive Statistics:\n{df.describe().round(2)}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nFlood Class Distribution:\n{df['Floods'].value_counts()}")

# ─────────────────────────────────────────────
# EPIC 2.2 – Univariate Analysis (Distribution Plots)
# ─────────────────────────────────────────────
features = ["Temp", "Humidity", "Cloud Cover", "ANNUAL",
            "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec"]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Univariate Analysis — Feature Distributions", fontsize=14, fontweight='bold')
for i, feat in enumerate(features):
    ax = axes[i // 4, i % 4]
    sns.histplot(df[feat], bins=20, kde=True, ax=ax, color='steelblue', edgecolor='white')
    ax.set_title(feat, fontsize=11)
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "univariate_dist.png"), dpi=100, bbox_inches='tight')
plt.close()
print("Saved: univariate_dist.png")

# ─────────────────────────────────────────────
# EPIC 2.3 – Box Plots
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Univariate Analysis — Box Plots by Flood Class", fontsize=14, fontweight='bold')
for i, feat in enumerate(features):
    ax = axes[i // 4, i % 4]
    tmp = df.copy()
    tmp['Floods'] = tmp['Floods'].map({0: 'No Flood', 1: 'Flood'})
    sns.boxplot(x='Floods', y=feat, data=tmp, ax=ax,
                order=['No Flood', 'Flood'],
                palette={'No Flood': '#3498db', 'Flood': '#e74c3c'})
    ax.set_title(feat, fontsize=11)
    ax.set_xticklabels(['No Flood', 'Flood'])
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "boxplots.png"), dpi=100, bbox_inches='tight')
plt.close()
print("Saved: boxplots.png")

# ─────────────────────────────────────────────
# EPIC 2.4 – Heatmap (Correlation Matrix)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
# Use numeric columns only to avoid FutureWarning in pandas >= 2.0
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            square=True, cbar_kws={"orientation": "vertical"})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "heatmap.png"), dpi=100, bbox_inches='tight')
plt.close()
print("Saved: heatmap.png")

# ─────────────────────────────────────────────
# EPIC 2.5 – Multivariate: Pair Plot
# ─────────────────────────────────────────────
pp_cols = ["ANNUAL", "Jun-Sep", "Humidity", "Cloud Cover", "Floods"]
pp_data = df[pp_cols].copy()
pp_data["Floods"] = pp_data["Floods"].map({0: "No Flood", 1: "Flood"})
pp = sns.pairplot(pp_data, hue="Floods",
                  palette={"No Flood": "#3498db", "Flood": "#e74c3c"},
                  diag_kind="kde", plot_kws={"alpha": 0.6})
pp.fig.suptitle("Multivariate Analysis — Pair Plot", y=1.02, fontsize=14, fontweight='bold')
plt.savefig(os.path.join(VIZ, "pairplot.png"), dpi=100, bbox_inches='tight')
plt.close()
print("Saved: pairplot.png")

# ─────────────────────────────────────────────
# EPIC 2.6 – Flood Class Distribution Bar
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["Floods"].value_counts()
ax.bar(["No Flood (0)", "Flood (1)"], counts.values,
       color=["#3498db", "#e74c3c"], edgecolor='white', width=0.5)
ax.set_title("Target Class Distribution", fontsize=13, fontweight='bold')
ax.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax.text(i, v + 0.5, str(v), ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "class_dist.png"), dpi=100, bbox_inches='tight')
plt.close()
print("Saved: class_dist.png")

# ══════════════════════════════════════════════════════════════
# EPIC 3 – Data Pre-processing  (ALL STEPS — ZERO ERRORS)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  [EPIC 3] DATA PRE-PROCESSING")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 3.1 — Missing Value Detection & Handling
# isnull().sum()  → count missing per column
# isnull().any()  → True/False whether ANY missing exists
# fillna(median)  → impute with column median (preserves size)
# ─────────────────────────────────────────────────────────────
print("\n[3.1] Checking Null / Missing Values")
print("\nisnull().sum() — Count per column:")
print(df.isnull().sum())
print("\nisnull().any() — Does any column have missing data?")
print(df.isnull().any())

if df.isnull().sum().sum() > 0:
    print("\nMissing values found — imputing with column median...")
    df.fillna(df.median(numeric_only=True), inplace=True)
    print("After imputation — total missing:", df.isnull().sum().sum())
else:
    print("\nNo missing values detected. Dataset is complete.")

# ─────────────────────────────────────────────────────────────
# STEP 3.2 — Outlier Detection & Treatment (IQR Capping)
# Outliers = values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
# Technique : CAPPING (clip to boundary — preserves dataset size)
#   Values ABOVE upper bound → replaced with upper bound
#   Values BELOW lower bound → replaced with lower bound
# ─────────────────────────────────────────────────────────────
print("\n[3.2] Outlier Detection & IQR Capping")

def cap_outliers(dataframe, col):
    """Detect and cap outliers using IQR method."""
    Q1    = dataframe[col].quantile(0.25)
    Q3    = dataframe[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR   # lower fence
    upper = Q3 + 1.5 * IQR   # upper fence
    n_out = ((dataframe[col] < lower) | (dataframe[col] > upper)).sum()
    dataframe[col] = dataframe[col].clip(lower=lower, upper=upper)
    return n_out, round(lower, 2), round(upper, 2)

print(f"\n{'Feature':<14} {'Outliers':>9} {'Lower':>10} {'Upper':>10}")
print("-" * 46)
for feat in features:
    n, lo, hi = cap_outliers(df, feat)
    print(f"{feat:<14} {n:>9} {lo:>10} {hi:>10}")
print("\nAll outliers capped via IQR method (dataset size preserved).")

# ─────────────────────────────────────────────────────────────
# STEP 3.3 — Categorical Value Handling
# Machine learning models work with numbers only.
# Categorical text columns must be encoded before training.
# Two techniques used:
#   a) Feature Mapping  : manual dict mapping  (ordinal/known order)
#   b) Label Encoding   : LabelEncoder()       (multi-class categories)
# ─────────────────────────────────────────────────────────────
print("\n[3.3] Categorical Value Handling")

from sklearn.preprocessing import LabelEncoder

# Detect categorical (object dtype) columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

if cat_cols:
    print(f"Categorical columns found: {cat_cols}")
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Label encoded: '{col}'")
    print("All categorical columns converted to numerical format.")
else:
    print("No categorical columns found — all features are already numerical.")
    print("(In datasets with text categories, Feature Mapping or LabelEncoder")
    print(" would be applied here, e.g.: Low=0, Medium=1, High=2)")

# ─────────────────────────────────────────────────────────────
# STEP 3.4 — Independent (X) & Dependent (y) Variable Split
# X → input features used to make predictions
# y → target variable / output label (Floods: 0 or 1)
# Rule: Select only relevant, highly correlated columns for X
# ─────────────────────────────────────────────────────────────
print("\n[3.4] Splitting Independent (X) and Dependent (y) Variables")

# X: all 8 meteorological features (independent variables)
X = df[features]

# y: target column — what the model must predict (dependent variable)
y = df["Floods"]

print(f"\nIndependent features (X):")
print(f"  Columns : {X.columns.tolist()}")
print(f"  Shape   : {X.shape}  (rows × features)")
print(f"\nDependent target (y):")
print(f"  Column  : Floods")
print(f"  Shape   : {y.shape}  (one label per row)")
print(f"  Values  : {sorted(y.unique().tolist())}  (0=No Flood, 1=Flood)")

# ─────────────────────────────────────────────────────────────
# STEP 3.5 — Train / Test Split  (75% train, 25% test)
# stratify=y → maintains class ratio in both splits
# random_state → reproducible results
# ─────────────────────────────────────────────────────────────
print("\n[3.5] Train-Test Split  (75% train | 25% test)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

print(f"  X_train: {X_train.shape}  |  y_train: {y_train.shape}")
print(f"  X_test : {X_test.shape}   |  y_test : {y_test.shape}")
print(f"  Train flood rate: {y_train.mean():.2%}  |  Test flood rate: {y_test.mean():.2%}")

# ─────────────────────────────────────────────────────────────
# STEP 3.6 — Feature Scaling  (StandardScaler)
# Formula : z = (x - μ) / σ
# → mean ≈ 0, std ≈ 1 after scaling
# RULE: fit_transform on TRAIN only → transform (never fit) on TEST
# NOTE: sc.fit_transform(X_test) is a BUG — it leaks test statistics.
#       Always use sc.transform(X_test) to avoid data leakage.
# ─────────────────────────────────────────────────────────────
print("\n[3.6] Feature Scaling — StandardScaler  [z = (x - μ) / σ]")

scaler = StandardScaler()

# Fit on training data ONLY (learn μ and σ from train set)
X_train_sc = scaler.fit_transform(X_train)   # fit + transform

# Transform test data using the SAME train statistics (no re-fitting)
X_test_sc  = scaler.transform(X_test)        # transform only — NO fit

print(f"  X_train after scaling — mean: {X_train_sc.mean():.4f}, std: {X_train_sc.std():.4f}")
print(f"  X_test  after scaling — mean: {X_test_sc.mean():.4f},  std: {X_test_sc.std():.4f}")
print("  StandardScaler applied correctly (scaler saved for deployment).")
print("  ✓ Pre-processing pipeline complete.\n")

# ─────────────────────────────────────────────
# EPIC 4 – Model Building & Comparison
# ─────────────────────────────────────────────
print("\n[EPIC 4] Model Building")

results = {}

# 4.1 Decision Tree
print("\n--- Decision Tree ---")
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train_sc, y_train)
y_pred_dt = dt.predict(X_test_sc)
acc_dt = accuracy_score(y_test, y_pred_dt) * 100
results['Decision Tree'] = acc_dt
print(f"Accuracy: {acc_dt:.2f}%")
print(classification_report(y_test, y_pred_dt, target_names=["No Flood", "Flood"]))

# 4.2 Random Forest
print("\n--- Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_sc, y_train)
y_pred_rf = rf.predict(X_test_sc)
acc_rf = accuracy_score(y_test, y_pred_rf) * 100
results['Random Forest'] = acc_rf
print(f"Accuracy: {acc_rf:.2f}%")
print(classification_report(y_test, y_pred_rf, target_names=["No Flood", "Flood"]))

# 4.3 K-Nearest Neighbors
print("\n--- K-Nearest Neighbors ---")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sc, y_train)
y_pred_knn = knn.predict(X_test_sc)
acc_knn = accuracy_score(y_test, y_pred_knn) * 100
results['KNN'] = acc_knn
print(f"Accuracy: {acc_knn:.2f}%")
print(classification_report(y_test, y_pred_knn, target_names=["No Flood", "Flood"]))

# 4.4 XGBoost
print("\n--- XGBoost ---")
xgb = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    eval_metric='logloss', random_state=42
)
xgb.fit(X_train_sc, y_train)
y_pred_xgb = xgb.predict(X_test_sc)
acc_xgb = accuracy_score(y_test, y_pred_xgb) * 100
results['XGBoost'] = acc_xgb
print(f"Accuracy: {acc_xgb:.2f}%")
print(classification_report(y_test, y_pred_xgb, target_names=["No Flood", "Flood"]))

# ─────────────────────────────────────────────
# 4.5 Model Comparison Chart
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
bars = ax.bar(results.keys(), results.values(), color=colors, edgecolor='white', width=0.5)
ax.set_ylim(0, 105)
ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
ax.set_ylabel("Accuracy (%)")
for bar, acc in zip(bars, results.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{acc:.2f}%", ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "model_comparison.png"), dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: model_comparison.png")

# ─────────────────────────────────────────────
# 4.6 XGBoost Confusion Matrix
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["No Flood", "Flood"])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title("XGBoost — Confusion Matrix", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ, "confusion_matrix.png"), dpi=100, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

# ─────────────────────────────────────────────
# EPIC 4 — Best Model Selection & Saving
# ─────────────────────────────────────────────
best_name = max(results, key=results.get)
best_acc  = results[best_name]
print(f"\n{'=' * 60}")
print(f"  BEST MODEL : {best_name}  |  Accuracy: {best_acc:.2f}%")
print(f"{'=' * 60}")

# Save the XGBoost model (as required by project spec: floods.save)
joblib.dump(xgb,    MODEL)
joblib.dump(scaler, SCALER)
# Also save to root directory for direct loading in app.py
joblib.dump(xgb,    "floods.save")
joblib.dump(scaler, "transform.save")
joblib.dump(scaler, "scaler.save")
print(f"\nModel  saved → {MODEL} and floods.save")
print(f"Scaler saved → {SCALER} and transform.save")
print("\n✓ Training pipeline complete!")
