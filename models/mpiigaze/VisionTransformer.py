import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

print("Libraries imported successfully.")

# -------------------------------
# FeedForward block
# -------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# Attention block
# -------------------------------
class Attention(nn.Module):
    def __init__(self, dim, n_head, dropout, head_dim):
        super().__init__()
        self.inner_dim = n_head * head_dim
        self.scale = head_dim ** -0.5
        self.n_head = n_head
        self.head_dim = head_dim
        self.dropout = dropout

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3)

        # Re-Attention weights
        self.reattn_weights = nn.Parameter(torch.randn(n_head, head_dim, head_dim))

        self.out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), qkv)

        # Attention computation
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = nn.Dropout(self.dropout)(attn)

        # Re-Attention
        attn = torch.einsum('b h i j, h d e -> b h i j', attn, self.reattn_weights)

        # Aggregate
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out(out)
        return out

# -------------------------------
# Transformer Block
# -------------------------------
class Transformer(nn.Module):
    def __init__(self, dim, depth, n_head, head_dim, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Attention(dim, n_head, dropout, head_dim),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        for norm, attn, ffn in self.layers:
            x = attn(norm(x)) + x
            x = ffn(x) + x
        return x

# -------------------------------
# DeepViT Architecture
# -------------------------------
class DeepViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, "pool type must be either 'cls' or 'mean'"

        self.to_patch_embeddings = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embeddings(img)
        b, n, _ = x.shape

        cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)
        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

# -------------------------------
# Instantiate Model
# -------------------------------
model = DeepViT(
    image_size=224,
    patch_size=16,
    num_classes=2,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

print(model)
