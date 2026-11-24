"""
Student Stress – final analysis (Session 12)
CSC1171 Data Analytics & Data Mining | Dataset: StressLevelDataset.csv (Kaggle)

Research question
    Which group of factors – psychological, physiological, environmental,
    academic or social – is most strongly associated with student stress level?

Methods (all covered in lectures)
    Supervised:   logistic regression, decision tree, random forest, k-nearest neighbours
    Unsupervised: PCA, k-means clustering
    Evaluation:   train/test split, 5-fold cross-validation, accuracy,
                  precision / recall / F1, confusion matrices, baselines

Pipeline
    0. Load and check the data
    1. Train/test split (80/20, stratified). The test set is not touched until step 4.
    2. Baselines: majority class, and the one-line blood pressure rule
    3. Tune and compare four classifiers with 5-fold CV on the training set
    4. Evaluate every tuned model once on the test set
    5. Interpret: logistic regression coefficients, random forest importances,
       a shallow decision tree
    6. Answer the research question: one model per factor group
    7. PCA: how many dimensions does the data really have?
    8. k-means: do natural clusters match the stress labels?
    9. Error analysis: where do the models go wrong?

Blood pressure is excluded from all models (see Presentation 1: it almost
encodes the label, so including it would inflate accuracy for the wrong reason).

Run:  python stress_final.py StressLevelDataset.csv
Outputs (tables, charts, results.json) are written to ./out_final
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, cross_val_score, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

SEED = 42
CSV = sys.argv[1] if len(sys.argv) > 1 else "StressLevelDataset.csv"
OUT = Path("out_final")
OUT.mkdir(exist_ok=True)
R = {}                                   # every number used in the slides

# chart style
INK, INK2, MUTED, GRID = "#1b1f2a", "#52514e", "#898781", "#e6e5e1"
STRESS_COLORS = ["#86b6ef", "#2a78d6", "#104281"]
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK2, "savefig.dpi": 200, "figure.facecolor": "white"})

GROUPS = {
    "Psychological": ["anxiety_level", "self_esteem", "mental_health_history", "depression"],
    "Physiological": ["headache", "sleep_quality", "breathing_problem"],          # BP excluded
    "Environmental": ["noise_level", "living_conditions", "safety", "basic_needs"],
    "Academic": ["academic_performance", "study_load", "teacher_student_relationship",
                 "future_career_concerns"],
    "Social": ["social_support", "peer_pressure", "extracurricular_activities", "bullying"],
}

# ======================================================== 0. load and check
df = pd.read_csv(CSV)
TARGET = "stress_level"
assert df.isna().sum().sum() == 0 and df.duplicated().sum() == 0
y = df[TARGET]
X = df.drop(columns=[TARGET])
FEATURES = [c for c in X.columns if c != "blood_pressure"]          # 19 features
print(f"Rows: {len(df)} | features used: {len(FEATURES)} | classes: {y.value_counts().sort_index().to_dict()}")

# ======================================================== 1. split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
R["n_train"], R["n_test"] = len(X_train), len(X_test)

# ======================================================== 2. baselines
majority = y_train.mode()[0]
R["baseline_majority"] = accuracy_score(y_test, np.full(len(y_test), majority))
BP_RULE = {1: 1, 2: 0, 3: 2}
R["baseline_bp_rule"] = accuracy_score(y_test, X_test["blood_pressure"].map(BP_RULE))
print(f"\nBaselines on test set: majority class {R['baseline_majority']:.3f} | "
      f"blood pressure rule {R['baseline_bp_rule']:.3f}")

# ======================================================== 3. tune and compare (training set only)
# Scaling sits inside each pipeline, so it is re-fitted on every CV fold (no leakage).
candidates = {
    "Logistic regression": (
        Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=4000))]),
        {"model__C": [0.01, 0.1, 1, 10]}),
    "Decision tree": (
        Pipeline([("model", DecisionTreeClassifier(random_state=SEED))]),
        {"model__max_depth": [2, 3, 4, 5, 6, 8, None], "model__min_samples_leaf": [1, 5, 10, 20]}),
    "Random forest": (
        Pipeline([("model", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]),
        {"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 5]}),
    "k-nearest neighbours": (
        Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier())]),
        {"model__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31]}),
}
tuned, rows = {}, []
for name, (pipe, grid) in candidates.items():
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train[FEATURES], y_train)
    tuned[name] = gs.best_estimator_
    i = gs.best_index_
    folds = [gs.cv_results_[f"split{k}_test_score"][i] for k in range(5)]
    rows.append({"Model": name,
                 "Best settings": ", ".join(f"{k.split('__')[1]}={v}" for k, v in gs.best_params_.items()),
                 "CV accuracy": np.mean(folds), "CV sd": np.std(folds)})
    if name == "k-nearest neighbours":
        knn_curve = pd.DataFrame({"k": [p["model__n_neighbors"] for p in gs.cv_results_["params"]],
                                  "cv_acc": gs.cv_results_["mean_test_score"]})
cv_table = pd.DataFrame(rows)
print("\nStep 3 – 5-fold CV on training set (19 features, no blood pressure):")
print(cv_table.round(3).to_string(index=False))
knn_curve.round(4).to_csv(OUT / "knn_k_curve.csv", index=False)
R["knn_curve"] = knn_curve.round(4).to_dict("list")

# ======================================================== 4. test-set evaluation (once)
test_rows, cms = [], {}
for name, model in tuned.items():
    pred = model.predict(X_test[FEATURES])
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro")
    test_rows.append({"Model": name, "Test accuracy": accuracy_score(y_test, pred),
                      "Macro precision": p, "Macro recall": r, "Macro F1": f})
    cms[name] = confusion_matrix(y_test, pred, labels=[0, 1, 2])
results = cv_table.merge(pd.DataFrame(test_rows), on="Model")
results.round(3).to_csv(OUT / "model_results.csv", index=False)
print("\nStep 4 – test set (220 students, used once):")
print(results.drop(columns="Best settings").round(3).to_string(index=False))
R["models"] = results.round(4).to_dict("records")

best_name = results.sort_values("CV accuracy", ascending=False).iloc[0]["Model"]
# Logistic regression is reported as the main model: its CV score is within one
# standard deviation of the best, and its coefficients can be interpreted directly.
MAIN = "Logistic regression"
R["main_model"] = MAIN
R["cm_main"] = cms[MAIN].tolist()
R["cms"] = {k: v.tolist() for k, v in cms.items()}
print(f"\nHighest CV accuracy: {best_name}. Main (interpretable) model: {MAIN}")
print(classification_report(y_test, tuned[MAIN].predict(X_test[FEATURES]), digits=3))

# ======================================================== 5. interpretation
# 5a. logistic regression coefficients, high-stress class vs low-stress class
lr = tuned[MAIN].named_steps["model"]
coef = pd.Series(lr.coef_[2] - lr.coef_[0], index=FEATURES).sort_values()
coef.round(3).to_csv(OUT / "lr_coefficients_high_vs_low.csv")
R["lr_coef_high_vs_low"] = coef.round(3).to_dict()

# 5b. random forest impurity-based importance
rf = tuned["Random forest"].named_steps["model"]
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
imp.round(4).to_csv(OUT / "rf_feature_importance.csv")
R["rf_importance"] = imp.round(4).to_dict()
print("Top random forest importances:\n", imp.head(6).round(3))

# 5c. a shallow tree people can read
small_tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_train[FEATURES], y_train)
R["small_tree_test_acc"] = accuracy_score(y_test, small_tree.predict(X_test[FEATURES]))
fig, ax = plt.subplots(figsize=(16, 6.4))
nodes = plot_tree(small_tree, feature_names=[f.replace("_", " ") for f in FEATURES],
                  class_names=["Low", "Medium", "High"], filled=False, rounded=True,
                  impurity=False, proportion=True, precision=2, fontsize=11, ax=ax)
# colour each box by its predicted stress level (same blues as the other charts)
fills = {"Low": "#d6e6fa", "Medium": "#86b6ef", "High": "#2a78d6"}
for node in nodes:
    txt = node.get_text()
    for lab, colr in fills.items():
        if f"class = {lab}" in txt:
            node.get_bbox_patch().set_facecolor(colr)
            node.get_bbox_patch().set_edgecolor("#52514e")
            if lab == "High":
                node.set_color("white")
fig.savefig(OUT / "decision_tree_depth3.png", bbox_inches="tight", pad_inches=0.1); plt.close(fig)
print(f"Depth-3 tree test accuracy: {R['small_tree_test_acc']:.3f}")

# ======================================================== 6. research question: one model per group
grp_rows = []
for g, cols in GROUPS.items():
    for name in ["Logistic regression", "Random forest"]:
        model = candidates[name][0]
        s = cross_val_score(model, X_train[cols], y_train, cv=cv, scoring="accuracy")
        grp_rows.append({"Group": g, "Model": name, "Features": len(cols),
                         "CV accuracy": s.mean(), "CV sd": s.std()})
s = cross_val_score(candidates[MAIN][0], X_train[FEATURES], y_train, cv=cv)
grp_rows.append({"Group": "All 19 features", "Model": MAIN, "Features": 19,
                 "CV accuracy": s.mean(), "CV sd": s.std()})
grp = pd.DataFrame(grp_rows)
grp.round(4).to_csv(OUT / "group_models.csv", index=False)
print("\nStep 6 – one model per factor group (5-fold CV, training set):")
print(grp.round(3).to_string(index=False))
R["group_models"] = grp.round(4).to_dict("records")

# ======================================================== 7. PCA
Z = StandardScaler().fit_transform(X[FEATURES])
pca = PCA().fit(Z)
evr = pca.explained_variance_ratio_
R["pca_evr"] = evr.round(4).tolist()
R["pca_pc1_loadings"] = pd.Series(pca.components_[0], index=FEATURES).round(3).to_dict()
pc1_model = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=1)),
                      ("model", LogisticRegression(max_iter=4000))])
R["pc1_only_cv"] = cross_val_score(pc1_model, X_train[FEATURES], y_train, cv=cv).mean()
print(f"\nStep 7 – PC1 explains {evr[0]:.1%}; PC2 {evr[1]:.1%}. "
      f"Logistic regression on PC1 alone: CV accuracy {R['pc1_only_cv']:.3f}")

scores = pca.transform(Z)[:, :2]
fig, ax = plt.subplots(figsize=(8, 6))
for s_, c, lab in zip([0, 1, 2], STRESS_COLORS, ["Low", "Medium", "High"]):
    m = y.values == s_
    ax.scatter(scores[m, 0], scores[m, 1], s=14, color=c, label=lab,
               edgecolor="white", linewidth=0.3)
ax.set_xlabel(f"PC1 ({evr[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
ax.legend(title="Stress level", frameon=False)
ax.grid(color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "pca_scatter.png"); plt.close(fig)

# ======================================================== 8. k-means
inertia = {k: KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Z).inertia_ for k in range(1, 9)}
R["kmeans_inertia"] = {str(k): round(v) for k, v in inertia.items()}
clusters = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit_predict(Z)
ct = pd.crosstab(clusters, y, rownames=["cluster"], colnames=["stress"])
# match each cluster to its majority stress level
matched = sum(ct.max(axis=1))
R["kmeans_crosstab"] = ct.values.tolist()
R["kmeans_match"] = matched / len(y)
print("\nStep 8 – k-means inertia:", R["kmeans_inertia"])
print(ct)
print(f"Clusters agree with the stress label for {R['kmeans_match']:.1%} of students "
      "(each cluster mapped to its majority level)")

# ======================================================== 9. error analysis
# Out-of-fold predictions for every student, so each one is predicted by a model
# that never saw it.
oof = cross_val_predict(candidates[MAIN][0], X[FEATURES], y, cv=cv)
wrong = oof != y.values
odd = ((X["blood_pressure"] == 3) & (y < 2)).values      # BP 3 but labelled low/medium
R["odd_rows"] = int(odd.sum())
R["errors_total"] = int(wrong.sum())
R["errors_in_odd"] = int((wrong & odd).sum())
R["acc_odd"] = float(1 - wrong[odd].mean())
R["acc_rest"] = float(1 - wrong[~odd].mean())
prof = df.assign(group=np.where(odd, "Unusual rows", "Other rows")) \
    .groupby(["group", TARGET])[["anxiety_level", "self_esteem", "depression", "bullying"]].mean().round(1)
prof.to_csv(OUT / "error_profiles.csv")
R["profiles"] = {f"{g}|{s}": v for (g, s), v in prof.to_dict("index").items()}
print(f"\nStep 9 – {R['odd_rows']} students have BP 3 but low/medium stress.")
print(f"  They hold {R['errors_in_odd']} of the {R['errors_total']} errors.")
print(f"  Accuracy on them: {R['acc_odd']:.1%} | on the other {len(y) - R['odd_rows']}: {R['acc_rest']:.1%}")
print("Factor averages: unusual rows look like medium-stress students whatever their label")
print(prof)

with open(OUT / "results.json", "w") as f:
    json.dump(R, f, indent=2, default=float)
print("\nAll outputs saved to", OUT.resolve())
