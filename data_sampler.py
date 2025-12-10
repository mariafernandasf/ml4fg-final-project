import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def get_unique_label(series):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    # check if there's >1 unique label
    if s.nunique() > 1:
        print("Warning: inconsistent labels for one individual:", s.unique())
    
    # just use the first label for that individual
    return s.iloc[0]


class CellSequenceDataset(Dataset):
    def __init__(
        self,
        adata,
        pca_key="X_pca",  
        sample_key="sampleID",
        seq_len=64,          # total cells per sequence (L)
        samples_per_epoch=100_000,
    ):
        """
        adata: AnnData containing only the individuals for this split
        seq_len:         int, length of each cell sequence 
        samples_per_epoch: how many sequences we treat as one 'epoch'
        """
        self.seq_len = seq_len
        self.samples_per_epoch = samples_per_epoch

        # 1) PCA matrix (n_cells, d_pca)
        X_pca = adata.obsm[pca_key]
        if not isinstance(X_pca, np.ndarray):
            X_pca = np.asarray(
                X_pca.todense() if hasattr(X_pca, "todense") else X_pca
            )
        self.X = X_pca  # (n_cells, d_pca)
        self.d_pca = self.X.shape[1]

        # 2) Sample IDs per cell
        sample_ids = np.array(adata.obs[sample_key])
        self.sample_ids = sample_ids

        # 3) Map individual to array of cell indices
        self.indiv2idx = {}
        for sid in np.unique(sample_ids):
            idxs = np.where(sample_ids == sid)[0]
            self.indiv2idx[sid] = idxs

        # Only keep individuals with enough cells
        self.indiv_ids = [
            sid for sid, idxs in self.indiv2idx.items()
            if len(idxs) >= seq_len
        ]
        if len(self.indiv_ids) == 0:
            raise ValueError("No individuals have at least seq_len cells.")

        # Choose sampling probability per individual (proportional to # of cells)
        indiv_sizes = np.array([len(self.indiv2idx[sid]) for sid in self.indiv_ids])
        self.indiv_probs = indiv_sizes / indiv_sizes.sum()

    def __len__(self):
        # how many sequences we draw per epoch
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 1) Sample an individual
        sid = np.random.choice(self.indiv_ids, p=self.indiv_probs)
        cell_indices = self.indiv2idx[sid]

        # 2) Sample seq_len distinct cells
        chosen = np.random.choice(cell_indices, size=self.seq_len, replace=False)

        # 3) Random permutation
        np.random.shuffle(chosen)

        # 4) Get PCA vectors with dim (L x d_pca)
        seq = self.X[chosen]

        # 5) input and target
        x_in = torch.from_numpy(seq[:-1]).float()    # (L-1, d_pca)
        x_target = torch.from_numpy(seq[1:]).float()  # (L-1, d_pca)

        return {
            "x_in": x_in,
            "x_target": x_target,
            "sample_id": sid,
        }
    
def get_dataloaders(adata_train, 
                 adata_val, 
                 adata_test, 
                 seq_len,
                 samples_per_epoch=10_000,
                pca_key = "X_pca",
                sample_key = "sampleID",
                ):
    train_dataset = CellSequenceDataset(
        adata_train,
        pca_key=pca_key,
        sample_key=sample_key,
        seq_len=seq_len,
        samples_per_epoch=100_000,  
    )

    val_dataset = CellSequenceDataset(
        adata_val,
        pca_key=pca_key,
        sample_key=sample_key,
        seq_len=seq_len,
        samples_per_epoch=10_000,
    )

    test_dataset = CellSequenceDataset(
        adata_test,
        pca_key=pca_key,
        sample_key=sample_key,
        seq_len=seq_len,
        samples_per_epoch=10_000,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,    # randomness is inside CellSequenceDataset
        num_workers=4,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader