import torch
import wandb 
import numpy as np 

def gmm_nll(x_next, pi, mu, var):
    """
    x_next: (B, T, d)
    pi:     (B, T, K)
    mu:     (B, T, K, d)
    var:    (B, T, K, d)
    returns scalar NLL
    """
    B, T, d = x_next.shape
    K = pi.size(-1)

    x_exp = x_next.unsqueeze(2)  # (B, T, 1, d)

    # log N(x | mu, var) with diagonal covariance
    log_norm_const = -0.5 * (d * torch.log(torch.tensor(2.0 * torch.pi, device=x_next.device)) +
                             torch.log(var).sum(-1))         # (B, T, K)
    quad = -0.5 * (((x_exp - mu) ** 2) / var).sum(-1)        # (B, T, K)
    log_prob = log_norm_const + quad                         # (B, T, K)

    log_pi = torch.log(pi + 1e-8)                            # (B, T, K)
    log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)     # (B, T)

    nll = -log_mix.mean()
    return nll

def train_one_epoch(model, loader, optimizer, device, epoch, scheduler, log_every=100):
    model.train()
    total_loss = 0.0
    n_batches = 0
    global_step = epoch * len(loader)


    for i, batch in enumerate(loader):
        x_in = batch["x_in"].to(device)         # (B, L-1, d_pca)
        x_target = batch["x_target"].to(device) # (B, L-1, d_pca)

        optimizer.zero_grad()
        pi, mu, var, _ = model(x_in)           # outputs for each position

        loss = gmm_nll(x_target, pi, mu, var)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        step = global_step + i
        if (i + 1) % log_every == 0:
            wandb.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/step": step,
                    "train/lr": scheduler.get_last_lr()[0],
                },
                step=step,
            )

    return total_loss / max(n_batches, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x_in = batch["x_in"].to(device)
        x_target = batch["x_target"].to(device)

        pi, mu, var, _ = model(x_in)
        loss = gmm_nll(x_target, pi, mu, var)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


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

    # optional: normalize to unit length (common for embeddings)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # store in AnnData
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
