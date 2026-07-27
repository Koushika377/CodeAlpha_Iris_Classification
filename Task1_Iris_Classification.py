"""
CodeAlpha Data Science Internship
TASK 1: Iris Flower Classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42

# 1. LOAD DATA
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(df.head(), "\n")
print(df.describe(), "\n")
print("Class balance:\n", df["species"].value_counts(), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")

# 2. EXPLORATORY DATA ANALYSIS (EDA)
sns.set_style("whitegrid")

pair = sns.pairplot(df, hue="species", diag_kind="hist", palette="Set2")
pair.fig.suptitle("Iris Feature Relationships by Species", y=1.02)
pair.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.close("all")

plt.figure(figsize=(6, 5))
sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("iris_correlation_heatmap.png", dpi=150)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.flatten(), iris.feature_names):
    sns.boxplot(data=df, x="species", y=col, hue="species", palette="Set2", ax=ax, legend=False)
    ax.set_title(f"{col} by species")
plt.tight_layout()
plt.savefig("iris_boxplots.png", dpi=150)
plt.close()

print("EDA plots saved: iris_pairplot.png, iris_correlation_heatmap.png, iris_boxplots.png\n")

# 3. PREPROCESSING
X = df[iris.feature_names]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. TRAIN MULTIPLE MODELS & COMPARE
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

results = {}
print("=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5)
    results[name] = {"model": model, "test_acc": acc, "cv_mean": cv_scores.mean()}
    print(f"\n--- {name} ---")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(classification_report(y_test, preds))

# 5. PICK BEST MODEL & DETAILED EVALUATION
best_name = max(results, key=lambda k: results[k]["test_acc"])
best_model = results[best_name]["model"]
print("=" * 60)
print(f"BEST MODEL: {best_name} (Test Accuracy: {results[best_name]['test_acc']:.4f})")
print("=" * 60)

best_preds = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, best_preds, labels=iris.target_names)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=150)
plt.close()

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=iris.feature_names)
    plt.figure(figsize=(7, 4))
    importances.sort_values().plot(kind="barh", color="teal")
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig("iris_feature_importance.png", dpi=150)
    plt.close()
    print("\nFeature importance plot saved: iris_feature_importance.png")

plt.figure(figsize=(8, 5))
comp_df = pd.DataFrame({k: [v["test_acc"], v["cv_mean"]] for k, v in results.items()},
                        index=["Test Accuracy", "CV Mean Accuracy"]).T
comp_df.plot(kind="bar", ax=plt.gca(), color=["#4C72B0", "#55A868"])
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.ylim(0.85, 1.02)
plt.tight_layout()
plt.savefig("iris_model_comparison.png", dpi=150)
plt.close()

print("\nAll plots saved. Confusion matrix: iris_confusion_matrix.png")
print("Model comparison chart: iris_model_comparison.png")
print("\nDone.")