import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import LabelEncoder
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def aggregate_pca_by_individual(adata, pca_key="X_pca", agg="mean"):
    X = adata.obsm[pca_key]
    sample_ids = adata.obs["sampleID"].values

    individuals = np.unique(sample_ids)
    aggregated = []
    labels = []

    for person in individuals:
        idx = np.where(sample_ids == person)[0]
        person_cells = X[idx]

        if agg == "mean":
            vec = person_cells.mean(axis=0)
        elif agg == "median":
            vec = np.median(person_cells, axis=0)
        elif agg == "mean_var":
            vec = np.concatenate([person_cells.mean(axis=0),
                                  person_cells.var(axis=0)])
        else:
            raise ValueError("Unknown agg")

        aggregated.append(vec)
        labels.append(person)

    return np.array(aggregated), np.array(labels)


def covid_embeddings_visualize(adata, embs, emb_labels):
    reducer = umap.UMAP(random_state=42)
    umap_coords = reducer.fit_transform(embs)

    covid_labels = adata.obs.set_index("sampleID")["covid_status"].to_dict()
    covid_colors = np.array([covid_labels[l] for l in emb_labels])

    plt.figure(figsize=(7,7))
    plt.scatter(umap_coords[:,0], umap_coords[:,1], c=(covid_colors=="positive"), cmap="coolwarm", s=10)
    plt.title("Sequence-level embedding UMAP")
    plt.show()


    covid_labels = adata.obs.set_index("sampleID")["covid_severity"].to_dict()
    severity = np.array([covid_labels[l] for l in emb_labels])

    categories = ["control", "mild/moderate", "severe/critical"]
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    severity_int = np.array([cat_to_int[c] for c in severity])
    colors = ["#bdbdbd", "#1f77b4", "#d62728"]    # gray, blue, red
    cmap = ListedColormap(colors)

    plt.figure(figsize=(7, 7))
    plt.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=severity_int,
        cmap=cmap,
        s=10
    )
    plt.title("Sequence-level embedding UMAP (COVID severity)")
    plt.xticks([])
    plt.yticks([])

    # Add legend using the category labels
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=colors[i], markersize=8)
            for i in range(len(categories))]
    plt.legend(handles, categories, title="COVID severity", loc="best")

    plt.show()


def covid_severity_prediction(adata, emb_labels, embs):
    # Run on all individuals instead of just test set
    pca_feats, _ = aggregate_pca_by_individual(adata)

    # Average sequence embeddings per person
    unique_ids = np.unique(emb_labels)
    model_agg = []

    for uid in unique_ids:
        idx = np.where(emb_labels == uid)[0]
        model_agg.append(embs[idx].mean(axis=0))

    model_feats = np.array(model_agg)

    severity_per_individual = (
        adata.obs.groupby("sampleID")["covid_severity"]
        .first()      # If consistent inside individual
        .loc[unique_ids]
        .values
    )

    # Encode 3-class severity label
    le = LabelEncoder()
    y_sev = le.fit_transform(severity_per_individual)   # 0/1/2

    # Train-test split
    X_train_pca, X_val_pca, y_train_sev, y_val_sev = train_test_split(
        pca_feats, y_sev, test_size=0.2, stratify=y_sev, random_state=42
    )

    X_train_mod, X_val_mod, _, _ = train_test_split(
        model_feats, y_sev, test_size=0.2, stratify=y_sev, random_state=42
    )

    # Multi-class logistic regression
    clf_pca = LogisticRegression(max_iter=500, multi_class="multinomial", solver="lbfgs")
    clf_mod = LogisticRegression(max_iter=500, multi_class="multinomial", solver="lbfgs")

    clf_pca.fit(X_train_pca, y_train_sev)
    clf_mod.fit(X_train_mod, y_train_sev)

    proba_pca = clf_pca.predict_proba(X_val_pca)       # (n,3)
    proba_mod = clf_mod.predict_proba(X_val_mod)

    # Multiclass AUC (macro-OVR)
    auc_pca_sev = roc_auc_score(y_val_sev, proba_pca, multi_class="ovr", average="macro")
    auc_mod_sev = roc_auc_score(y_val_sev, proba_mod, multi_class="ovr", average="macro")

    print("=== SEVERITY (Multiclass) ===")
    print("PCA   ROC-AUC:", auc_pca_sev)
    print("MODEL ROC-AUC:", auc_mod_sev)

    # plot ROC/PR curves for 'severe' class
    sev_class_index = list(le.classes_).index("severe/critical")
    plot_roc_pr_curves(
        y_val_sev == sev_class_index,
        proba_pca[:, sev_class_index],
        "Severity (PCA) – Severe vs Rest"
    )

    plot_roc_pr_curves(
        y_val_sev == sev_class_index,
        proba_mod[:, sev_class_index],
        "Severity (Model) – Severe vs Rest"
    )

def plot_roc_pr_curves(y_true, y_score, title_prefix):
    # ---------- ROC ----------
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # ---------- PR ----------
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(1, 2, figsize=(12,4))

    # ROC
    ax[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax[0].plot([0,1],[0,1],'--', color="gray")
    ax[0].set_title(f"{title_prefix} – ROC Curve")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].legend()

    # PR Curve
    ax[1].plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    base_rate = np.mean(y_true)
    ax[1].hlines(base_rate, 0, 1, color='gray', linestyle="--", label=f"Baseline = {base_rate:.3f}")
    ax[1].set_title(f"{title_prefix} – Precision–Recall Curve")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].legend()

    plt.show()

    return roc_auc, pr_auc