"""
For now, All models take:
- image of shape (B, 1, 28, 28)
- action of shape (B, 3) where action = (log_scale, dx, dy)
and output:
- next_image of shape (B, 1, 28, 28)
"""

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


#================================================
#                SIMPLE MLP
#================================================
class SimpleMLP(nn.Module):
    """MLP that predicts the next glimpse image from (image, action).

    Args:
        img_hw: Spatial size of the (square) input image.
        action_dim: Size of the action vector (log_scale, dx, dy -> 3).
        hidden_dim: Width of the hidden Linear layers.
    """
    def __init__(self, img_hw: int = 28, action_dim: int = 3, hidden_dim: int = 512):
        super().__init__()
        self.img_hw = img_hw
        self.img_dim = img_hw * img_hw

        self.net = nn.Sequential(
            nn.Linear(self.img_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.img_dim),
            nn.Sigmoid(), # Final activation (sigmoid) so pixel values stay in [0, 1].
        )

    def forward(self, images, actions, ar_steps=0):
        """
        Args:
            images: (B, T, C, H, W)
            actions: (B, T, A)
        """
        B = images.size(0)
        T = images.size(1)
        # x = images.view(B, T, self.img_dim) # Flatten the input image to a vector of size 28*28 = 784.
        images_flat = rearrange(images, "b t c h w -> b t (c h w)") # Flatten the input image to a vector of size 1*28*28 = 784.
        
        if not ar_steps or ar_steps==0:
            x = images_flat
            x = torch.cat([x, actions], dim=-1)
            x = self.net(x) # (B, T, 28* 28)
            x = x.view(B, T, self.img_hw, self.img_hw)
            return x

        # Run through mlp model, auto-regressively up until ar_steps
        preds = []
        ar_steps = min(ar_steps, T)
        x = images_flat[:, 0] # the seed image
        for t in range(ar_steps):
            x = torch.cat([x, actions[:, t]], dim=-1) #  Concatenate with the action vector of size 3, giving an input of size 787.
            x = self.net(x) # (B, 1, 28, 28)
            preds.append(x.view(B, 1, self.img_hw, self.img_hw))
        
        # Change from list to torch tensor of shape (B, T, H, W)
        preds_ar = torch.stack(preds, dim=1).squeeze(2)
        
        if ar_steps >= T:
            return preds_ar
        
        # Tail: teacher-forced on the remaining true frames, conditioned on the AR prefix
        x_full = torch.cat([preds_ar, images_flat[:, ar_steps+1]], dim=1)
        preds_tail = torch.cat([x_full, actions])[:, ar_steps:]
        
        return torch.cat([preds_ar, preds_tail], dim=1)



#================================================
#                SIMPLE CNN
#================================================
class SimpleCNN(nn.Module):
    """CNN that predicts the next glimpse image from (image, action).

    Args:
        img_hw: Spatial size of the (square) input image. Must be divisible by 4.
        action_dim: Size of the action vector (log_scale, dx, dy -> 3).
        base_channels: Channels of the first conv layer (doubled at each downsample).
        hidden_dim: Width of the fused feature vector that conditions the decoder.
    """
    def __init__(self, img_hw: int = 28, action_dim: int = 3, base_channels: int = 32, hidden_dim: int = 256):
        super().__init__()
        assert img_hw % 4 == 0, f"img_hw must be divisible by 4, got {img_hw}"
        self.img_hw = img_hw
        self.feat_hw = img_hw // 4 # Two stride-2 downsamples
        self.feat_c  = base_channels * 2
        self.feat_dim = self.feat_c * self.feat_hw * self.feat_hw

        # Encoder: 28 -> 14 -> 7
        self.encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.to_feat   = nn.Linear(self.feat_dim, hidden_dim)
        self.action_fc = nn.Linear(action_dim, hidden_dim)
        self.from_feat = nn.Linear(hidden_dim, self.feat_dim)

        # Decoder: 7 -> 14 -> 28
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]

        # Encode image to a feature vector
        h = self.encoder(image) # (B, 2C, H/4, W/4)
        h = h.view(B, self.feat_dim)
        img_feat = self.to_feat(h) # (B, hidden_dim)

        # Fuse with action by addition
        a_feat = self.action_fc(action) # (B, hidden_dim)
        fused = img_feat + a_feat

        # Decode back to image
        h = self.from_feat(fused).view(B, self.feat_c, self.feat_hw, self.feat_hw)
        out = self.decoder(h)
        return out


#================================================
#       VAE (CNN or MLP encoder/decoder)
#================================================
# Constructor takes a flag (e.g. backbone="cnn" or "mlp") to pick which encoder/decoder to use.
#
# Encoder:
# 1. If backbone is CNN, use the Simple CNN encoder stack to map image to a feature vector of size D.
# 2. If backbone is MLP, flatten the image and pass through Linear + ReLU layers to a feature vector of size D.
# 3. From that feature vector, two Linear heads produce mu (B, Z) and logvar (B, Z).
#
# Reparameterization:
# 1. Sample epsilon from a standard normal of shape (B, Z).
# 2. Compute z = mu + exp(0.5 * logvar) * epsilon.
#
# Action conditioning:
# 1. Project action (B, 3) through a small Linear layer to size Z (or some action embedding size).
# 2. Concatenate z and the action embedding into a vector of size Z + Z_action.
#
# Decoder:
# 1. If backbone is CNN, Linear from (Z + Z_action) up to C*H'*W', reshape, then ConvTranspose2d + ReLU stack back to (B, 1, 28, 28).
# 2. If backbone is MLP, Linear + ReLU layers from (Z + Z_action) up to 784 and reshape to (B, 1, 28, 28).
# 3. Final sigmoid so pixel values stay in [0, 1].
#
# Forward returns: predicted next_image, mu, logvar (so the train loop can compute reconstruction loss + KL divergence).


#================================================
#         My Custom LeWorldModel Modules
#================================================
class DeepMLP(nn.Module):
    """A simple MLP whose depth can be customized"""
    def __init__(self, input_dim, hidden_dim, output_dim, depth):
        super().__init__()
        
        # defining simple mlp blocks
        layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()]
        
        # Building our deep net
        for i in range(depth -1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()]
        layers += [nn.Linear(hidden_dim, output_dim)] # last layer
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
        

class ARPredictorSimple(nn.Module):
    """
    A Simple Auto-regressive predictor. Given action and current glimpse
    it outputs next expected glimpse.
    
    Uses a simple MLP instead of transformer.
    """
    def __init__(self, z_dim_img, z_dim_action, hidden_dim=512, depth=3):
        super().__init__()

        self.net = DeepMLP(z_dim_img + z_dim_action, hidden_dim, z_dim_img, depth=depth)

    def forward(self, z_img, z_action):
        """
        Parallel pass; autoregressive rollout is handled by JEPA.predict.

        Args:
            z_img: (B, T, D_z)
            z_action: (B, T, D_a)
        """
        x = torch.cat([z_img, z_action], dim=-1)
        return self.net(x)

        
class ARPredictorSimpleAdaLN(nn.Module):
    """FiLM-conditioned predictor with optional autoregressive rollout."""
    def __init__(self, z_dim, a_dim, hidden_dim=512):
        super().__init__()
        self.up = nn.Linear(z_dim, hidden_dim)
        self.film = nn.Linear(a_dim, 2 * hidden_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.down = nn.Linear(hidden_dim, z_dim)

    def _step(self, z, a):
        # per-step block: (B, *, D_z), (B, *, D_a) -> (B, *, D_z)
        h = self.up(z)
        gamma, beta = self.film(a).chunk(2, dim=-1)
        h = h * (1 + gamma) + beta
        return self.down(h)

    def forward(self, z_img, z_action):
        """
        Parallel pass; autoregressive rollout is handled by JEPA.predict.

        Args:
            z_img:    (B, T, D_z)
            z_action: (B, T, D_a)
        Returns:  (B, T, D_z)
        """
        return self._step(z_img, z_action)


# class ARPredictorLSTM(nn.Module):
#     """LSTM predictor with hidden state for memory across the trajectory.

#     Args:
#         z_dim_img:    dim of input/output image embedding
#         z_dim_action: dim of action embedding
#         hidden_dim:   LSTM hidden state size
#         num_layers:   stack depth (1 is fine for MNIST scale)
#     """
#     def __init__(self, z_dim_img, z_dim_action, hidden_dim=512, num_layers=1):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             input_size=z_dim_img + z_dim_action,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#         )
#         self.head = nn.Linear(hidden_dim, z_dim_img)

#     def forward(self, z_img, z_action, ar_steps=0):
#         """
#         z_img:    (B, T, D_img)
#         z_action: (B, T, D_action)
#         ar_steps: 0 = pure teacher forcing, T = pure AR, in-between = curriculum
#         Returns:  z_pred (B, T, D_img)
#         """
#         B, T, _ = z_img.size()

#         # Teacher-forced path — single parallel LSTM call. Fast and stable.
#         if ar_steps <= 0:
#             x = torch.cat([z_img, z_action], dim=-1)         # (B, T, D_in)
#             h, _ = self.lstm(x)                               # (B, T, H)
#             return self.head(h)                               # (B, T, D_img)

#         # Mixed path: AR for the first ar_steps, TF for the rest.
#         preds = []
#         x_t = z_img[:, 0:1]                                   # seed = true first frame
#         h_state = None                                        # LSTM zero-init h, c

#         for t in range(ar_steps):
#             inp = torch.cat([x_t, z_action[:, t:t+1]], dim=-1)   # (B, 1, D_in)
#             h, h_state = self.lstm(inp, h_state)                  # (B, 1, H), state carries
#             x_t = self.head(h)                                     # (B, 1, D_img) — predicted next
#             preds.append(x_t)

#         # Remaining steps teacher-forced, continuing from the carried hidden state.
#         if ar_steps < T:
#             tail_in = torch.cat([z_img[:, ar_steps:], z_action[:, ar_steps:]], dim=-1)
#             h, _ = self.lstm(tail_in, h_state)
#             preds.append(self.head(h))

#         return torch.cat(preds, dim=1)                         # (B, T, D_img)


# class ARPredictorTransformer(nn.Module):
#     """Causal transformer predictor. Each position t attends to ≤ t.

#     Args:
#         z_dim_img:    dim of input/output image embedding
#         z_dim_action: dim of action embedding
#         hidden_dim:   internal width
#         depth:        number of transformer layers
#         heads:        number of attention heads
#         mlp_mult:     FFN expansion factor
#         dropout:      dropout in attn + FFN
#         max_T:        max trajectory length (for positional embeddings)
#     """
#     def __init__(self, z_dim_img, z_dim_action, hidden_dim=256, depth=4,
#                  heads=4, mlp_mult=4, dropout=0.1, max_T=32):
#         super().__init__()
#         self.input_proj = nn.Linear(z_dim_img + z_dim_action, hidden_dim)
#         self.pos_emb = nn.Parameter(torch.zeros(1, max_T, hidden_dim))
#         nn.init.normal_(self.pos_emb, std=0.02)

#         layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=heads,
#             dim_feedforward=hidden_dim * mlp_mult,
#             dropout=dropout,
#             batch_first=True,
#             activation="gelu",
#             norm_first=True,        # pre-LN: more stable
#         )
#         self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
#         self.head = nn.Linear(hidden_dim, z_dim_img)

#     def _tf_forward(self, z_img, z_action):
#         """Teacher-forced parallel pass with causal mask."""
#         B, T, _ = z_img.size()
#         x = torch.cat([z_img, z_action], dim=-1)
#         x = self.input_proj(x) + self.pos_emb[:, :T]
#         mask = torch.triu(                                     # True = blocked
#             torch.ones(T, T, device=x.device, dtype=torch.bool),
#             diagonal=1,
#         )
#         h = self.transformer(x, mask=mask)
#         return self.head(h)

#     def forward(self, z_img, z_action, ar_steps=0):
#         """
#         ar_steps=0: pure teacher forcing (parallel, fast, stable)
#         ar_steps=T: pure AR
#         in between: AR for first ar_steps, TF for the rest
#         """
#         B, T, _ = z_img.size()
#         if ar_steps <= 0:
#             return self._tf_forward(z_img, z_action)

#         ar_steps = min(ar_steps, T)
#         z_in = z_img[:, 0:1]                                   # true seed
#         preds_ar = []
#         for t in range(ar_steps):
#             pred_seq = self._tf_forward(z_in, z_action[:, :t+1])
#             next_pred = pred_seq[:, -1:]                       # (B, 1, D)
#             preds_ar.append(next_pred)
#             if t + 1 < T:
#                 z_in = torch.cat([z_in, next_pred], dim=1)

#         preds_ar = torch.cat(preds_ar, dim=1)                  # (B, ar_steps, D)
#         if ar_steps >= T:
#             return preds_ar

#         # Tail with TF, conditioned on the AR-built prefix (full history available)
#         z_full = torch.cat([z_in, z_img[:, ar_steps:T]], dim=1)
#         full_preds = self._tf_forward(z_full, z_action)
#         preds_tail = full_preds[:, ar_steps:T]                 # (B, T-ar_steps, D)
#         return torch.cat([preds_ar, preds_tail], dim=1)        # (B, T, D)


class ImageEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, depth=3):
        super().__init__()
        
        self.encoder = DeepMLP(input_dim, hidden_dim, z_dim, depth=depth)

    def forward(self, images):
        """
        Args:
            images: (B*T, C, H, W)
        """
        images = rearrange(images, "b c h w -> b (c h w)")
        z_img = self.encoder(images)
        return z_img
    
class Decoder(nn.Module):
    """MLP decoder: z -> (1, H, W). Makes JEPA latents visualizable."""
    def __init__(self, z_dim, hidden_dim=512, h=28, w=28, depth=2):
        super().__init__()
        self.h = h
        self.w = w
        layers = [nn.Linear(z_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, h * w), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        # z: (..., D) -> (..., 1, H, W)
        x = self.net(z)
        return x.view(*z.shape[:-1], 1, self.h, self.w)

#================================================
#     ViT and JEPA modules from LeWorldModel
#================================================

class MLP_Projector(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)
    
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


def apply_rope(x, theta=10000.0):
    """
    Apply rotary position embeddings to q/k tensors.

    Args:
        x: (B, heads, T, dim_head)
    """
    dim = x.size(-1)
    rot_dim = dim - (dim % 2)
    if rot_dim == 0:
        return x

    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]

    positions = torch.arange(x.size(-2), device=x.device, dtype=x.dtype)
    freqs = torch.arange(0, rot_dim, 2, device=x.device, dtype=x.dtype)
    inv_freq = theta ** (-freqs / rot_dim)
    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]
    x_rot = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    x_rot = x_rot.flatten(-2)
    return torch.cat((x_rot, x_pass), dim=-1)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, rope_theta=10000.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        q = apply_rope(q, theta=self.rope_theta)
        k = apply_rope(k, theta=self.rope_theta)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)
    
class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, rope_theta=10000.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, rope_theta=rope_theta)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
        action_dim=None,
        rope_theta=10000.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        # If input dim  to predictor is different from the transformer's hidden_dim
        # we need to project input into the right dim hidden dimension size
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        # "condition" projector is for action; the action embedding has its own dim
        action_dim = action_dim or input_dim
        self.cond_proj = (
            nn.Linear(action_dim, hidden_dim)
            if action_dim != hidden_dim
            else nn.Identity()
        )

        # If output dim from predictor is different specified dim of 
        # the latent vector we expect to compare it to (here it is the output_dim)
        # then we need to create a projector too. 
        
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout, rope_theta=rope_theta)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            # If block layer is of class 'Block', it can only accept one input x
            # If it is of type 'ConditionalBlock', it takes x and c (c is condition or action in this case)
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x
    
    
def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class ConditionalBlock(nn.Module):
    """
    Transformer block with AdaLN-zero conditioning.    
    AdaLN stands for Adaptive Layer Normalization.    
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, rope_theta=10000.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, rope_theta=rope_theta)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
    
class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        action_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        rope_theta=10000.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
            action_dim=action_dim,
            rope_theta=rope_theta,
        )

    def forward(self, x, c):
        """
        Parallel causal pass. Autoregressive rollout is handled by JEPA.predict.

        Args:
            x: (B, T, d) encoded observations
            c: (B, T, act_dim) encoded actions
        """
        x = self.dropout(x)
        return self.transformer(x, c)
    
    
    
class MemoryPredictor(nn.Module):
    """Causal transformer that builds a running memory latent over a trajectory.

    Trains recurrence in parallel: at step t the causal self-attention over
    positions <= t is the accumulated prior memory, and a zero-content seed token
    (position 0, tagged with a learned separator embedding) is the initial memory.

    Note: for a literal recurrent m_{t-1} -> m_t feedback, swap the parallel pass
    for a sequential loop over T. Kept as a documented alternative, not used here.
    """

    def __init__(self, z_dim_img, z_dim_memory, hidden_dim=256, depth=2,
                 heads=4, dim_head=64, mlp_dim=512, rope_theta=10000.0):
        super().__init__()
        self.frame_proj = nn.Linear(z_dim_img, hidden_dim)
        # Learned segment embeddings separate the memory seed from frame tokens
        self.seg_mem = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.seg_frame = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.seg_mem, std=0.02)
        nn.init.normal_(self.seg_frame, std=0.02)

        # input already at hidden_dim, so input_proj is Identity; output maps to z_memory
        self.transformer = Transformer(
            hidden_dim, hidden_dim, z_dim_memory, depth, heads, dim_head,
            mlp_dim, block_class=Block, rope_theta=rope_theta,
        )

    def forward(self, z_img, z_memory_seed=None):
        """
        Args:
            z_img: (B, T, z_dim_img) encoded frame latents
            z_memory_seed: (B, hidden_dim) optional initial memory, defaults to zeros
        Returns:
            z_memory: (B, T, z_dim_memory) where z_memory[:, t] saw frames <= t
        """
        B = z_img.size(0)
        tokens = self.frame_proj(z_img) + self.seg_frame # (B, T, hidden)

        if z_memory_seed is None:
            seed = self.seg_mem.expand(B, 1, -1) # zero content + separator
        else:
            seed = z_memory_seed.unsqueeze(1) + self.seg_mem

        seq = torch.cat([seed, tokens], dim=1) # (B, T+1, hidden)
        out = self.transformer(seq) # causal pass
        return out[:, 1:] # drop seed position, keep per-frame memory


class ActionEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        smoothed_dim=None,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        in_dim = smoothed_dim or input_dim
        self.patch_embed = nn.Linear(input_dim, smoothed_dim) if smoothed_dim else nn.Identity()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )
        
    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = self.patch_embed(x)
        x = self.embed(x)
        return x


#================================================
#     Spatial-latent JEPA modules (ViT tokens)
#================================================
# Validated direction: a ViT token grid (e.g. 16 tokens x 8) instead of a flat vector keeps spatial
# layout, so the decoder renders sharp digits instead of blur. The predictor is spatiotemporal
# (block-causal AdaLN) and predicts token residuals. See SpatialJEPA in jepa.py.

class ViTBlock(nn.Module):
    """Pre-LN transformer block, full (bidirectional) self-attention."""
    def __init__(self, dim, heads=4, mlp=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp), nn.GELU(), nn.Linear(dim * mlp, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


class ViTSpatialEncoder(nn.Module):
    """Image -> (B, N, C) patch tokens. Per-token non-affine LN pins scale for SigReg."""
    def __init__(self, patch=7, c=8, hidden=64, depth=2, heads=4, img=28):
        super().__init__()
        self.p, self.np = patch, img // patch
        n = self.np ** 2
        self.embed = nn.Linear(patch * patch, hidden)
        self.pos = nn.Parameter(torch.randn(1, n, hidden) * 0.02)
        self.blocks = nn.ModuleList([ViTBlock(hidden, heads) for _ in range(depth)])
        self.to_latent = nn.Linear(hidden, c)
        self.norm = nn.LayerNorm(c, elementwise_affine=False)

    def forward(self, img):  # (B*,1,H,W) -> (B*,N,C)
        x = self.embed(rearrange(img, "b o (h p1) (w p2) -> b (h w) (o p1 p2)", p1=self.p, p2=self.p)) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(self.to_latent(x))


class ViTSpatialDecoder(nn.Module):
    """(B*, N, C) -> logits (B*,1,H,W). ViT blocks give tokens global context, then a CONV render head
    upsamples the token grid so neighbors blend (no per-patch seams/gridding)."""
    def __init__(self, patch=7, c=8, hidden=64, depth=2, heads=4, img=28):
        super().__init__()
        self.np = img // patch
        n = self.np ** 2
        self.from_latent = nn.Linear(c, hidden)
        self.pos = nn.Parameter(torch.randn(1, n, hidden) * 0.02)
        self.blocks = nn.ModuleList([ViTBlock(hidden, heads) for _ in range(depth)])
        if self.np == 4:    # patch 7: 4 -> 7 -> 14 -> 28
            self.head = nn.Sequential(
                nn.ConvTranspose2d(hidden, hidden, 4, 1, 0), nn.GELU(),
                nn.ConvTranspose2d(hidden, 32, 4, 2, 1), nn.GELU(),
                nn.ConvTranspose2d(32, 1, 4, 2, 1))
        elif self.np == 7:  # patch 4: 7 -> 14 -> 28
            self.head = nn.Sequential(
                nn.ConvTranspose2d(hidden, 32, 4, 2, 1), nn.GELU(),
                nn.ConvTranspose2d(32, 1, 4, 2, 1))
        else:
            raise ValueError(f"conv head supports img//patch in {{4,7}}, got {self.np}")

    def forward(self, tok):  # (B*,N,C) -> (B*,1,H,W) logits
        x = self.from_latent(tok) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = rearrange(x, "b (h w) d -> b d h w", h=self.np)
        return self.head(x)


class STBlock(nn.Module):
    """Spatiotemporal block with AdaLN-zero on the action. Tokens are flat (B, L, H); a block-causal
    mask lets frame t attend to all tokens of frames <= t."""
    def __init__(self, dim, heads=4, mlp=4):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp), nn.GELU(), nn.Linear(dim * mlp, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, mask):  # x:(B,L,H) c:(B,L,H) mask:(L,L)
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        h = self.n1(x) * (1 + sc1) + sh1
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b l (h d) -> b h l d", h=self.h) for t in (q, k, v))
        a = rearrange(F.scaled_dot_product_attention(q, k, v, attn_mask=mask), "b h l d -> b l (h d)")
        x = x + g1 * self.proj(a)
        x = x + g2 * self.mlp(self.n2(x) * (1 + sc2) + sh2)
        return x


class STPredictor(nn.Module):
    """Spatiotemporal residual predictor over a token grid. Block-causal across time, AdaLN on action;
    predicts the residual from the current frame's tokens (consecutive glimpses are close)."""
    def __init__(self, c=8, hidden=128, depth=4, heads=4, n=16, max_frames=16):
        super().__init__()
        self.n = n
        self.in_proj = nn.Linear(c, hidden)
        self.act_proj = nn.Linear(3, hidden)
        self.sp_pos = nn.Parameter(torch.randn(1, 1, n, hidden) * 0.02)
        self.tp_pos = nn.Parameter(torch.randn(1, max_frames, 1, hidden) * 0.02)
        self.blocks = nn.ModuleList([STBlock(hidden, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, c)

    def _mask(self, Tn, device):
        fi = torch.arange(Tn, device=device).repeat_interleave(self.n)
        return fi[None, :] <= fi[:, None]  # block-causal, True=keep

    def forward(self, tokens, action):  # (B,T,N,C),(B,T,3) -> next-frame tokens (B,T,N,C)
        B, Tn, Nn, _ = tokens.shape
        h = self.in_proj(tokens) + self.sp_pos + self.tp_pos[:, :Tn]
        h = rearrange(h, "b t n d -> b (t n) d")
        c = self.act_proj(action)[:, :, None, :].expand(B, Tn, Nn, -1)
        c = rearrange(c, "b t n d -> b (t n) d")
        mask = self._mask(Tn, tokens.device)
        for blk in self.blocks:
            h = blk(h, c, mask)
        delta = rearrange(self.out(self.norm(h)), "b (t n) c -> b t n c", t=Tn)
        return tokens + delta  # residual
