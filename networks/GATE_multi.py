import torch
import torch.nn as nn

from networks.base import BaseModel


class GaussianMaskedCrossAttention(nn.Module):
    def __init__(self, dim, epsilon=1e-8, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.epsilon = epsilon

    def forward(self, q, H, gaussian_mask=None):
        """
        q: (B, 1, D)  - cls token
        H: (B, Q, D)  - age query
        gaussian_mask: (B, Q)  - soft mask for each query slot
        """
        B, _, D = q.shape
        Qn = H.size(1)  # number of age queries
        head_dim = D // self.num_heads
        scale = head_dim ** -0.5

        # Linear projections
        Qh = self.q_proj(q).reshape(B, 1, self.num_heads, head_dim).transpose(1, 2)  # (B, heads, 1, head_dim)
        Kh = self.k_proj(H).reshape(B, Qn, self.num_heads, head_dim).transpose(1, 2)  # (B, heads, Qn, head_dim)
        Vh = self.v_proj(H).reshape(B, Qn, self.num_heads, head_dim).transpose(1, 2)  # (B, heads, Qn, head_dim)

        # Attention score: (B, heads, 1, Qn)
        attn_scores = torch.matmul(Qh, Kh.transpose(-2, -1)) * scale  # (B, heads, 1, Qn)

        if gaussian_mask is not None:
            # gaussian_mask: (B, Qn) -> (B, 1, 1, Qn) for broadcasting
            mask_logit = gaussian_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, Qn)
            attn_scores = attn_scores + torch.log(mask_logit+self.epsilon)

        # Softmax over query dimension
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)  # (B, heads, 1, Qn)

        # Weighted sum
        z = torch.matmul(attn_weights, Vh)  # (B, heads, 1, head_dim)
        z = z.transpose(1, 2).reshape(B, 1, D)  # (B, 1, D)

        # Residual connection & norm
        q = q + self.dropout(self.out_proj(z))
        q = self.norm(q)

        return q


class GATE_multi(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # backbone dimension
        if 'vitB16' in cfg.backbone:
            hdim = 512
        else:
            raise ValueError('Unsupported backbone')

        self.ref_points = torch.randn([cfg.ref_point_num, hdim])
        if cfg.start_norm:
            self.ref_points = nn.functional.normalize(self.ref_points, dim=-1)
        self.ref_points = nn.parameter.Parameter(self.ref_points)
        self.ref_point_num = cfg.ref_point_num

        nheads = 8
        self.num_layers = cfg.num_layers
        # Transformer decoder layer (self-attention + cross-attention + FFN)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hdim,
            nhead=nheads,
            dim_feedforward=2048,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )

        self.query_update_layers = cfg.query_update_layers
        self.transformer_decoder = nn.ModuleList([
            nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers) for _ in range(len(self.query_update_layers))
        ])

        self.gaussian_crossAttn = GaussianMaskedCrossAttention(dim=hdim, epsilon=cfg.epsilon, num_heads=nheads)

        # mu, sigma head
        self.mu_sigma_head = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.GELU(),
            nn.Linear(hdim, 2)
        )

        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max
        self.epsilon = cfg.epsilon

    def forward_features(self, images, dataset_idx):
        """
        images -> ViT + query 업데이트까지 한 features 반환
        """
        # --- ViT embedding ---
        x = self.encoder.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        ##################################################
        class_embeddings = self.encoder.class_embeddings[dataset_idx].to(x.dtype).unsqueeze(1)
        x = torch.cat(
            [class_embeddings + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        ######################################################
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # (L, B, D)

        # --- query ---
        B = x.size(1)
        query = self.ref_points.unsqueeze(0).repeat(B, 1, 1)  # (B, Q, D)


        j = 0
        # --- Transformer ---
        for i, blk in enumerate(self.encoder.transformer.resblocks):
            x = blk(x)  # patch token update
            if i in self.query_update_layers:
                H = x.permute(1, 0, 2)  # (B, N, D)
                H = self.encoder.ln_post(H)
                H = H @ self.encoder.proj
                h_patch = H[:, 1:]  # exclude CLS
                query = self.transformer_decoder[j](tgt=query, memory=h_patch)
                j += 1

        # --- final CLS token ---
        H_final = self.encoder.ln_post(x.permute(1, 0, 2))  # (B, N, D)
        if self.encoder.proj is not None:
            H_final = H_final @ self.encoder.proj

        return H_final, query

    def predict_mu_sigma(self, base_embs):
        # --- 1) CLS, patch seperation ---
        h_cls = base_embs[:, 0]  # CLS token

        # --- 2) μ, σ prediction ---
        mu_s = self.mu_sigma_head(h_cls)
        m, s = mu_s[:, 0], mu_s[:, 1]

        mu = torch.sigmoid(m) * self.ref_point_num
        sigma = torch.sigmoid(s) * self.sigma_max + self.sigma_min 
      
        return mu, sigma

    def forward_transformer(self, H, query, mu, sigma):
        h_cls = H[:, 0]
        Q = self.ref_point_num

        # gaussian mask for final aggregation
        with torch.no_grad():
            query_indices = torch.arange(Q, device=H.device).unsqueeze(0)  # (1, Q)
            mu_b, sigma_b = mu[:, None], sigma[:, None]

            gaussian_mask = torch.exp(- (query_indices - mu_b) ** 2 / (2 * sigma_b ** 2))  # (B, Q)
            gaussian_mask = gaussian_mask / (gaussian_mask.sum(dim=-1, keepdim=True) + self.epsilon)

        # gaussian cross-attention
        output = h_cls.unsqueeze(1)
        output = self.gaussian_crossAttn(output, query, gaussian_mask)

        return output.squeeze(1)

    def forward(self, images, dataset_idx):
        H, age_query = self.forward_features(images, dataset_idx)
        mu, sigma = self.predict_mu_sigma(H)
        features = self.forward_transformer(H, age_query, mu, sigma)

        return features
