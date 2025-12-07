import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoregressiveCellModel(nn.Module):
    def __init__(
        self,
        pca_dim: int = 50,     
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_hid: int = 1024,
        dropout: float = 0.1,
        n_components: int = 8,  # K in the GMM
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.pca_dim = pca_dim
        self.d_model = d_model
        self.n_components = n_components
        self.max_seq_len = max_seq_len

        # 1) Project PCA vector into model dimension
        self.input_proj = nn.Linear(pca_dim, d_model)

        # 2) Learned positional embeddings 
        #self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # 3) Transformer stack (decoder-style with causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True,  # (B, T, D)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # 4) GMM head (Mixture Density Network)
        self.fc_pi = nn.Linear(d_model, n_components)                # logits for mixture weights
        self.fc_mu = nn.Linear(d_model, n_components * pca_dim)      # means
        self.fc_logvar = nn.Linear(d_model, n_components * pca_dim)  # log-variances (diagonal)

    def _causal_mask(self, T: int, device):
        # [T, T] mask where True means "block attention"
        # We block attention to future positions (j > i)
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask  # used as src_mask in transformer

    def forward(self, x, src_key_padding_mask=None):
        """
        x: (B, T, pca_dim)  -- PCA vectors, teacher-forced
        src_key_padding_mask: (B, T) bool where True means "pad"
        Returns:
            pi:   (B, T, K)
            mu:   (B, T, K, pca_dim)
            var:  (B, T, K, pca_dim)
            h:    (B, T, d_model)  # transformer embeddings (for downstream)
        """
        B, T, D_in = x.shape
        assert D_in == self.pca_dim

        # 1) Input projection
        h = self.input_proj(x)  # (B, T, d_model)

        # 2) Add positional embeddings
        #positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        #h = h + self.pos_emb(positions)  # (B, T, d_model)

        # 3) Causal mask for autoregressive attention
        causal_mask = self._causal_mask(T, device=x.device)  # (T, T)

        # 4) Transformer
        # src_key_padding_mask: (B, T) with True at PAD positions
        h = self.transformer(
            h,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, T, d_model)

        # 5) GMM head
        # Mixture weights
        logits_pi = self.fc_pi(h)             # (B, T, K)
        pi = F.softmax(logits_pi, dim=-1)     # (B, T, K)

        # Means
        mu = self.fc_mu(h)                    # (B, T, K * pca_dim)
        mu = mu.view(B, T, self.n_components, self.pca_dim)  # (B, T, K, d)

        # Variances (diagonal)
        logvar = self.fc_logvar(h)            # (B, T, K * pca_dim)
        logvar = logvar.view(B, T, self.n_components, self.pca_dim)
        var = torch.exp(logvar)               # (B, T, K, d), elementwise >0

        return pi, mu, var, h

