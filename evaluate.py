import numpy as np
import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import scanpy as sc


def compute_model_embeddings(model, adata, pca_key="X_pca", batch_size=1024):
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

    adata.obsm["X_model"] = embeddings
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


def visualize_results(adata_test, model):

    compute_model_embeddings(model, adata_test, pca_key="X_pca")
    adata_test_copy1 = adata_test.copy()
    sc.pp.neighbors(adata_test_copy1, use_rep="X_pca")
    sc.tl.umap(adata_test_copy1, min_dist=0.3)
    adata_test_copy2 = adata_test.copy()
    sc.pp.neighbors(adata_test_copy2, use_rep="X_model")    
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

  
