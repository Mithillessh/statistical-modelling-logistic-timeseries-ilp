
"""
Part 1: Logistic Regression
Predicting Mouse (M/1) vs Raccoon (R/0) from measurement data
Dataset: measurements.csv (208 rows, 60 features V1-V60, target Class)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score, roc_curve)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =================================================================
# Q2: Data exploration
# =================================================================
df = pd.read_csv('data/measurements.csv')
print("Shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nClass counts:\n", df['Class'].value_counts())
print("\nClass proportion:\n", df['Class'].value_counts(normalize=True))

df['y'] = (df['Class'] == 'M').astype(int)

# Class distribution bar chart (Figure 1)
counts = df['Class'].map({'M': 'Mouse', 'R': 'Raccoon'}).value_counts()
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(counts.index, counts.values, color=['#4C72B0', '#DD8452'])
for i, v in enumerate(counts.values):
    ax.text(i, v + 1, str(v), ha='center', fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('Class Distribution: Mouse vs Raccoon')
plt.tight_layout()
plt.savefig('results/class_balance.png', dpi=150)
plt.show()

# =================================================================
# Q3.a: Split the data into training and testing
# =================================================================
# Reduced set of 12 predictors used instead of all 60 (selected beforehand; see README, Limitations)
selected = ['V1', 'V2', 'V5', 'V11', 'V12', 'V27', 'V28', 'V37', 'V48', 'V50', 'V52', 'V60']
X = df[selected]
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test, has_constant='add')
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit(disp=0)
print(result.summary())

# =================================================================
# Q3.b: Evaluation metrics
# =================================================================
pred_prob_test = result.predict(X_test_sm)
pred_class_test = (pred_prob_test > 0.5).astype(int)

acc = accuracy_score(y_test, pred_class_test)
prec = precision_score(y_test, pred_class_test)
rec = recall_score(y_test, pred_class_test)   # sensitivity (Mouse)
f1 = f1_score(y_test, pred_class_test)
auc = roc_auc_score(y_test, pred_prob_test)
cm = confusion_matrix(y_test, pred_class_test)

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)   # Raccoon

print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall (Sensitivity, Mouse): {rec:.4f}")
print(f"Test Specificity (Raccoon): {specificity:.4f}")
print(f"Test F1: {f1:.4f}")
print(f"Test AUC: {auc:.4f}")
print("Confusion matrix:\n", cm)

# ROC curve (Figure 2)
fpr, tpr, _ = roc_curve(y_test, pred_prob_test)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(fpr, tpr, color='#4C72B0', linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Test Set')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('results/roc_curve.png', dpi=150)
plt.show()

# Confusion matrix heatmap (Figure 3)
fig, ax = plt.subplots(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Raccoon', 'Mouse'], yticklabels=['Raccoon', 'Mouse'],
            ax=ax, cbar=False, annot_kws={'size': 16})
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — Test Set')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150)
plt.show()

# Omnibus test, Cox & Snell R2, Nagelkerke R2
n = len(y_train)
llf = result.llf
llnull = result.llnull

lr_chi2 = -2 * (llnull - llf)
r2_cs = 1 - np.exp((2/n) * (llnull - llf))
r2_nagel = r2_cs / (1 - np.exp((2/n) * llnull))

print(f"\nOmnibus LR chi-square: {lr_chi2:.2f}, df={result.df_model:.0f}, p={result.llr_pvalue:.2e}")
print(f"Cox & Snell R^2: {r2_cs:.3f}")
print(f"Nagelkerke R^2: {r2_nagel:.3f}")

# Classification table (training data, cut-off 0.5)
pred_prob_train = result.predict(X_train_sm)
pred_class_train = (pred_prob_train > 0.5).astype(int)
ct = pd.crosstab(y_train, pred_class_train)
print("\nClassification table (train):\n", ct)


def hosmer_lemeshow(y_true, y_prob, g=10):
    """Hosmer-Lemeshow goodness-of-fit test."""
    hl_df = pd.DataFrame({'y': y_true.values, 'prob': y_prob.values})
    hl_df['decile'] = pd.qcut(hl_df['prob'], g, duplicates='drop')
    grouped = hl_df.groupby('decile', observed=True)
    obs_events = grouped['y'].sum()
    obs_nonevents = grouped['y'].count() - obs_events
    exp_events = grouped['prob'].sum()
    exp_nonevents = grouped['prob'].count() - exp_events
    hl_stat = (((obs_events - exp_events) ** 2 / exp_events) +
               ((obs_nonevents - exp_nonevents) ** 2 / exp_nonevents)).sum()
    dof = len(grouped) - 2
    pval = 1 - stats.chi2.cdf(hl_stat, dof)
    return hl_stat, dof, pval


hl_stat, dof, pval = hosmer_lemeshow(y_train, pred_prob_train, g=10)
print(f"\nHosmer-Lemeshow chi-square: {hl_stat:.3f}, df={dof}, p={pval:.3f}")

# =================================================================
# Q4: Significance of each variable (Wald test)
# =================================================================
print("\nWald test p-values:\n", result.pvalues)

# =================================================================
# Q5: Odds ratios
# =================================================================
coefs = result.params
odds_ratios_1unit = np.exp(coefs)
odds_ratios_p01 = np.exp(coefs * 0.01)   # per +0.01 increase

print("\nOdds ratios (per 1-unit):\n", odds_ratios_1unit)
print("\nOdds ratios (per +0.01):\n", odds_ratios_p01)

# =================================================================
# Q6: Prediction for a new animal
# =================================================================
new_animal = pd.DataFrame([{
    'const': 1,
    'V1': 0.0162, 'V2': 0.0379, 'V5': 0.0344, 'V11': 0.3250, 'V12': 0.1316,
    'V27': 0.5844, 'V28': 0.9892, 'V37': 0.6729, 'V48': 0.0629,
    'V50': 0.0212, 'V52': 0.0079, 'V60': 0.0012
}])[X_train_sm.columns]

prob_mouse = result.predict(new_animal).iloc[0]
print(f"\nNew animal - predicted P(Mouse): {prob_mouse:.4f}")
print("Predicted class:", "Mouse (M)" if prob_mouse >= 0.5 else "Raccoon (R)")
