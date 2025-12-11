import torch
import wandb 
from models import AutoregressiveCellModel
import data_sampler
import time
import copy
from pathlib import Path

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

def train_model(
    adata_train,
    adata_val,
    adata_test,
    device="cuda",
    seq_len=64,
    pca_dim = 50, 
    d_model = 256,
    n_layers = 6,
    n_heads = 4,
    d_hid = 512,
    dropout = 0.1,
    n_components = 8,
    max_seq_len = 256,
    batch_size = 32,
    lr = 1e-4,
    epochs = 50,
    OUTPUT_DIR = Path("/home/ubuntu/ml4fg-final-project/output_models"),
    ):
       
    # initialize model
    model = AutoregressiveCellModel(
        pca_dim = pca_dim,
        d_model=d_model,
        n_layers = n_layers,
        n_heads = n_heads,
        d_hid = d_hid,
        dropout = dropout,
        n_components = n_components,
        max_seq_len = max_seq_len,
        )
    
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # initialize Wandb
    wandb.init(
    project="covid-cell-transformer-gmm",
    config={
        "pca_dim": pca_dim,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dim_feedforward": d_hid,
        "n_components": n_components,
        "lr": lr,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "epochs": epochs,
    },
    entity="ms7073-columbia-university"
)

    wandb.define_metric("train/epoch_loss", summary="min", step_metric="epoch")
    wandb.define_metric("val/epoch_loss", summary="min", step_metric="epoch")
    wandb.define_metric("test/loss", summary="min")
    wandb.watch(model, log = "gradients", log_freq=200)

    # get data loaders
    train_loader, val_loader, test_loader = data_sampler.get_dataloaders(adata_train, 
                 adata_val, 
                 adata_test, 
                 seq_len,
                 samples_per_epoch=10_000,
                pca_key = "X_pca",
                sample_key = "sampleID",
                )

    # train model 
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # 1) Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, scheduler)

        # 2) Validate
        val_loss = evaluate(model, val_loader, device)

        scheduler.step()

        elapsed = time.time() - epoch_start_time
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"train loss {train_loss:5.4f} | val loss {val_loss:5.4f}"
        )

        # 3) Track best model by val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"New best model at epoch {best_epoch} with val loss={best_val_loss:5.4f}")

        
        wandb.log(
            {
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            },
        )

    print(f"Training complete. Best val loss {best_val_loss:5.4f} at epoch {best_epoch}")
    # save the best model
    torch.save(best_model_state, OUTPUT_DIR / f"best_model_{epochs}epochs_{seq_len}seqlen.pt")

    return test_loader, best_model_state