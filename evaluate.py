import numpy as np
import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def compute_model_embeddings(model, adata, pca_key="X_pca", batch_size=1024, model_key="X_model"):
    model.eval()
    device = next(model.parameters()).device

    # Get PCA matrix
    X_pca = adata.obsm[pca_key]
    if not isinstance(X_pca, np.ndarray):
        X_pca = np.asarray(X_pca)

    n_cells, pca_dim = X_pca.shape

    # infer embedding dim from model
    with torch.no_grad():
        dummy = torch.from_numpy(X_pca[:1]).float().unsqueeze(1).to(device)
        _, _, _, h = model(dummy)
        d_model = h.shape[-1]

    embeddings = np.zeros((n_cells, d_model), dtype=np.float32)

    # batch inference
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        batch = torch.from_numpy(X_pca[start:end]).float().to(device)
        batch = batch.unsqueeze(1)  # (B, 1, pca_dim)

        with torch.no_grad():
            pi, mu, var, h = model(batch)  # h: (B, 1, d_model)

        emb = h[:, 0, :].cpu().numpy()
        embeddings[start:end] = emb

    # normalize to unit length 
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    adata.obsm[model_key] = embeddings
    return embeddings


def compute_sequence_embeddings(
    model,
    adata,
    seq_len=128,
    samples_per_individual=5,
    pca_key="X_pca",
):
    model.eval()
    device = next(model.parameters()).device

    X_pca = adata.obsm[pca_key]
    if not isinstance(X_pca, np.ndarray):
        X_pca = np.asarray(X_pca)

    sample_ids = adata.obs["sampleID"].values
    individuals = np.unique(sample_ids)

    sequence_embeddings = []
    seq_labels = []

    for person in individuals:
        cell_idx = np.where(sample_ids == person)[0]
        person_cells = X_pca[cell_idx]

        for _ in range(samples_per_individual):
            # sample a sequence of t cells
            if len(person_cells) >= seq_len:
                idx = np.random.choice(len(person_cells), size=seq_len, replace=False)
            else:
                idx = np.random.choice(len(person_cells), size=seq_len, replace=True)

            seq = person_cells[idx]
            seq = torch.from_numpy(seq).float().unsqueeze(0).to(device)  # (1, t, d)

            with torch.no_grad():
                pi, mu, var, h = model(seq)  # h: (1, t, d_model)

            # final token embedding
            final_emb = h[:, -1, :].squeeze(0).cpu().numpy()
            sequence_embeddings.append(final_emb)
            seq_labels.append(person)

    sequence_embeddings = np.array(sequence_embeddings)
    seq_labels = np.array(seq_labels)
    return sequence_embeddings, seq_labels


def visualize_results(adata_test, model, model_key="X_model"):

    compute_model_embeddings(model, adata_test, pca_key="X_pca", model_key=model_key)
    adata_test_copy1 = adata_test.copy()
    sc.pp.neighbors(adata_test_copy1, use_rep="X_pca")
    sc.tl.umap(adata_test_copy1, min_dist=0.3)
    adata_test_copy2 = adata_test.copy()
    sc.pp.neighbors(adata_test_copy2, use_rep=model_key)    
    sc.tl.umap(adata_test_copy2, min_dist=0.3)  

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # embeddings UMAP by COVID status
    sc.pl.umap(adata_test_copy1, color="covid_status", ax=axes[0], title="PCA UMAP", show=False)
    sc.pl.umap(adata_test_copy2, color="covid_status", ax=axes[1], title="Model Embedding UMAP", show=False)
    plt.tight_layout()
    plt.show()  

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # embeddings UMAP by cell type
    sc.pl.umap(adata_test_copy1, color="celltype", ax=axes[0], title="PCA UMAP", show=False, legend_loc=None)
    sc.pl.umap(adata_test_copy2, color="celltype", ax=axes[1], title="Model Embedding UMAP", show=False, legend_loc=None)
    plt.tight_layout()
    plt.show()  

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # embeddings UMAP by COVID severity
    sc.pl.umap(adata_test_copy1, color="covid_severity", ax=axes[0], title="PCA UMAP", show=False)
    sc.pl.umap(adata_test_copy2, color="covid_severity", ax=axes[1], title="Model Embedding UMAP", show=False)
    plt.tight_layout()
    plt.show() 

    # compute silhouette scores for UMAP
    sil_pca_status   = compute_silhouette(adata_test_copy1, "covid_status",   rep_key="X_umap")
    sil_model_status = compute_silhouette(adata_test_copy2, "covid_status",   rep_key="X_umap")

    print(f"Silhouette (UMAP of PCA) by covid_status      : {sil_pca_status:.4f}")
    print(f"Silhouette (UMAP of model embedding) by status: {sil_model_status:.4f}")

    sil_pca_sev   = compute_silhouette(adata_test_copy1, "covid_severity", rep_key="X_umap")
    sil_model_sev = compute_silhouette(adata_test_copy2, "covid_severity", rep_key="X_umap")

    print(f"Silhouette (UMAP of PCA) by covid_severity      : {sil_pca_sev:.4f}")
    print(f"Silhouette (UMAP of model embedding) by severity: {sil_model_sev:.4f}")
    
    # compute silhouette scores for full embeddings
    sil_pca_status_full = compute_silhouette(adata_test, "covid_status", rep_key="X_pca")
    sil_model_status_full = compute_silhouette(adata_test, "covid_status", rep_key=model_key)

    print(f"Silhouette in PCA space by covid_status      : {sil_pca_status_full:.4f}")
    print(f"Silhouette in model embedding space by status: {sil_model_status_full:.4f}")

    sil_pca_sev_full = compute_silhouette(adata_test, "covid_severity", rep_key="X_pca")
    sil_model_sev_full = compute_silhouette(adata_test, "covid_severity", rep_key=model_key)

    print(f"Silhouette in PCA space by covid_severity      : {sil_pca_sev_full:.4f}")
    print(f"Silhouette in model embedding space by severity: {sil_model_sev_full:.4f}")


def compute_silhouette(adata, label_key, rep_key="X_umap"):
    X = adata.obsm[rep_key]
    labels = adata.obs[label_key]

    # Drop NaNs if any
    mask = ~pd.isna(labels)
    X = X[mask]
    labels = labels[mask]

    # Encode labels as integers (silhouette_score wants numeric labels)
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels.astype(str))

    return silhouette_score(X, labels_enc, metric="euclidean")

