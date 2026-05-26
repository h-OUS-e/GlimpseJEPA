"""
A script to quickly train an ML model and test things quickly without config. 
Keeps it flexible for experimenting fast
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from einops import rearrange
import matplotlib.pyplot as plt

from glimpse import rollout
from jepa import JEPA
from ml_layers import ARPredictorSimple, ActionEncoder, ImageEncoder, Decoder
from vis_utils import plot_glimpse_frames


#================================================
#                GLOBAL VARS
#================================================
scale_sensitivity = 0.2
translation_sensitivity = 0.1
batch_size = 64
T_max = 16
lr = 1e-3
epochs = 20
viz_every = 1
lambd = 0.01 # sigreg loss coefficient
lambd_recon = 0.01

# model params
input_dim_action = 3 # log_scale, x, y are only 3 parameters
hidden_dim_img_encoder = 512
hidden_dim_predictor = 512
decoder_hidden_dim = 512
z_dim_img = 36
z_dim_action = 9
depth_img_encoder = 3
depth_predictor = 3


# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#================================================
#                DATASET PREP
#================================================
# 1. Get MNIST batch of shape (B, 28, 28) for train and validation. Shuffle them.
train_ds = torchvision.datasets.MNIST(root='./dataset', train=True,  download=True, transform=transforms.ToTensor())
val_ds   = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transforms.ToTensor())

# 2. Wrap them in DataLoaders so we can iterate batch by batch each epoch.
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)

# 3. Get a single Batch output and its shape
sample = next(iter(train_loader))[0]
if len(sample.shape) == 4:
    B, C, H, W = sample.size()
    
else:
    raise f"error, image has a a size that doesn't look like B, C, H, W. Input size is {len(sample.shape)}"

#================================================
#                MODEL AND OPTIMIZER
#================================================
# # 1. Define a model that takes a glimpse image (B, 1, 28, 28) and an action (B, 3) and outputs the next glimpse image (B, 1, 28, 28).
# model = SimpleMLP(img_hw=28, action_dim=3, hidden_dim=512)

# 1. Define JEPA & its model parts
image_encoder = ImageEncoder(H*W, hidden_dim_img_encoder, z_dim_img, depth=depth_img_encoder)
action_encoder = ActionEncoder(input_dim_action, emb_dim=z_dim_action)
predictor = ARPredictorSimple(z_dim_img, z_dim_action, hidden_dim_predictor, depth_predictor)
decoder = Decoder(z_dim=z_dim_img, hidden_dim=decoder_hidden_dim, h=H, w=W, depth=2)

model = JEPA(image_encoder, predictor, action_encoder, decoder=decoder)

# 2. Move the model to the right device (cuda if available, else cpu).
model = model.to(device)

# 3. Define an Adam optimizer over the model parameters with learning rate lr.
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# 4. Define an MSE loss to compare predicted glimpse against the true next glimpse.
loss_fn = nn.MSELoss()


#================================================
#                TRAIN LOOP
#================================================
def autoregressive_forward(seed_glimpse, actions):
    """Chain T_max model steps, feeding each prediction back as the next input."""
    current_glimpse_state = seed_glimpse
    preds = []
    
    for t in range(T_max):
        current_glimpse_state = model(current_glimpse_state, actions[t]) # (B, 1, 28, 28)
        preds.append(current_glimpse_state)
    return torch.stack(preds, dim=0) # (T_max, B, 1, 28, 28)


for epoch in tqdm(range(epochs), desc="epochs"):
    # ---- train ----
    model.train()
    train_loss_sum = 0.0
    train_batches = 0


    train_pbar = tqdm(train_loader, desc=f"train {epoch}", leave=False)
    for imgs, _ in train_pbar:
        # 1. Move batch to device, drop channel dim for Glimpse (B, 28, 28)
        imgs = imgs.to(device).squeeze(1)

        # 2-5. Roll out a random glimpse trajectory (no grad through glimpse rendering)
        with torch.no_grad():
            seed, actions, input_images, target_images = rollout(imgs, T_max, scale_sensitivity, translation_sensitivity, device=device)
            
        # 6. Inference
        ar_steps = min(epoch+1, T_max) # grow horizon over training
        z_preds, z_img, z_action = model(input_images, actions, ar_steps=ar_steps)

        # # 7. MSE over all T_max predicted frames vs true frames
        # loss = loss_fn(preds, target_images)
        
        # 7. JEPA loss
        # encode target images
        z_targets, _ = model.encode(target_images)
        # get mse loss
        loss_mse = model.mse_last_step(z_preds, z_targets)
        # get sigreg loss
        loss_sigreg = model.sigreg_loss(z_img)
        # recon loss on encoder's latents
        # loss_recon  = model.recon_loss(z_img, input_images) # or  model.recon_loss(z_targets, target_images)
        loss_recon = model.recon_loss(z_img.detach(), input_images) # Use this if you don't want recon loss to effect predictor or encoder weights
        # get total loss
        loss = loss_mse + lambd * loss_sigreg + lambd_recon * loss_recon

        # 8. Backprop through the whole chain
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 9. Track running loss
        train_loss_sum += loss.item()
        train_batches  += 1
        train_pbar.set_postfix(loss=f"{loss.item():.4f}")

    # ---- validation ----
    model.eval()
    val_loss_sum, val_batches = 0.0, 0
    val_pbar = tqdm(val_loader, desc=f"val {epoch}", leave=False)
    with torch.no_grad():
        for imgs, _ in val_pbar:
            imgs = imgs.to(device).squeeze(1)
            seed, actions, input_images, target_images = rollout(imgs, T_max, scale_sensitivity, translation_sensitivity, device=device)
            z_preds, z_img, z_action = model(input_images, actions, ar_steps=ar_steps)
            
            # JEPA loss
            z_targets, _ = model.encode(target_images)
            loss_mse = model.mse_last_step(z_preds, z_targets)
            loss_sigreg = model.sigreg_loss(z_img)
            loss = loss_mse + lambd * loss_sigreg
            val_loss_sum += loss.item()
            val_batches  += 1

    train_avg = train_loss_sum / max(train_batches, 1)
    val_avg   = val_loss_sum   / max(val_batches, 1)
    print(f"epoch {epoch:4d}  train {train_avg:.4f}  val {val_avg:.4f}")

    # Every N epochs, render true vs predicted glimpse sequences from the last val batch
    if epoch % viz_every == 0:
        true_seq = target_images.squeeze(2).cpu() # (B, T_max, 28, 28)
        z_preds_img = rearrange(z_preds, "b t (h w) -> b t h w", h=6, w=6) # Reshape latent vector to an image (just for viz)
        pred_seq = z_preds_img.squeeze(2).detach().cpu() # (B, T_max, 28, 28)
        
        # Reconstruct image from predictor and encoder
        recon_images_from_predictor = model.decode(z_preds).squeeze(2).detach().cpu()
        recon_images_from_encoder = model.decode(z_img).squeeze(2).detach().cpu()
        
        plot_glimpse_frames(true_seq)
        plot_glimpse_frames(recon_images_from_encoder)
        plot_glimpse_frames(recon_images_from_predictor)
        plot_glimpse_frames(pred_seq)
                
        # Collapse detector
        with torch.no_grad():
            z_flat = z_img.reshape(-1, z_img.size(-1))    # (N, D)
            z_std  = z_flat.std(0).mean().item()          # ~0 = collapsed, ~1 = healthy
            z_norm = z_flat.norm(dim=-1).mean().item()
            # pairwise cosine of random pairs — should not be ~1
            a, b = z_flat[:50], z_flat[50:100]
            cos = F.cosine_similarity(a, b).mean().item()
        print(f"std={z_std:.3f}  norm={z_norm:.3f}  cos={cos:.3f}")




#===========================================================================
#                            VALIDATION
#===========================================================================

# ============================================================
# Inverse-dynamics probe: predict action a_t from (z_t, z_{t+1})
# Frozen encoder + tiny MLP. Visualize predictions vs targets unrolled.
# Given a pair of consecutive embeddings (z_t, z_{t+1}), train an MLP 
# to recover the action a_t that produced the transition. If the encoder's 
# embeddings carry action-relevant info, the probe will succeed.
# ============================================================


device = next(model.parameters()).device
model.eval()

# -- 1. Collect (z_pair, action) pairs from frozen encoder --
def collect_pairs(n_batches=20):
    Z_pairs, A_targets = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            imgs, _ = next(iter(train_loader))
            imgs = imgs.squeeze(1).to(device)
            seed, actions, input_images, target_images = rollout(
                imgs, T_max, scale_sensitivity, translation_sensitivity, device=device,
            )
            z_in,  _ = model.encode(input_images)    # (B, T, D)  state at t
            z_tgt, _ = model.encode(target_images)   # (B, T, D)  state at t+1
            z_pair = torch.cat([z_in, z_tgt], dim=-1)  # (B, T, 2D)

            Z_pairs.append(z_pair)
            A_targets.append(actions)
    return torch.cat(Z_pairs, 0), torch.cat(A_targets, 0)   # both (N, T, *)

Z_pairs, A_tgt = collect_pairs(n_batches=20)
print("Z_pairs:", tuple(Z_pairs.shape), "  A_tgt:", tuple(A_tgt.shape))

# -- 2. Flatten across batch+time for training, train/val split --
B, T = Z_pairs.shape[:2]
X = Z_pairs.reshape(-1, Z_pairs.size(-1))    # (B*T, 2D)
y = A_tgt.reshape(-1, 3)                      # (B*T, 3)

perm = torch.randperm(X.size(0))
n_train = int(0.8 * X.size(0))
X_tr, X_va = X[perm[:n_train]], X[perm[n_train:]]
y_tr, y_va = y[perm[:n_train]], y[perm[n_train:]]

# -- 3. Tiny MLP probe --
probe = nn.Sequential(
    nn.Linear(X.size(-1), 64),
    nn.ReLU(),
    nn.Linear(64, 3),
).to(device)
opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

for step in range(2000):
    pred = probe(X_tr)
    loss = F.mse_loss(pred, y_tr)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        with torch.no_grad():
            val = F.mse_loss(probe(X_va), y_va).item()
        print(f"step {step:4d}  train {loss.item():.4f}  val {val:.4f}")

# -- 4. Unrolled prediction plot --
probe.eval()
with torch.no_grad():
    pred_actions = probe(Z_pairs.reshape(-1, Z_pairs.size(-1))).reshape(B, T, 3)

n_traj = 3
names = ["log_scale", "Δx", "Δy"]
fig, axes = plt.subplots(n_traj, 3, figsize=(12, 2.5 * n_traj), sharex=True)
t_axis = torch.arange(T).cpu()
for b in range(n_traj):
    for d in range(3):
        ax = axes[b, d]
        ax.plot(t_axis, A_tgt[b, :, d].cpu(), label="target", lw=2)
        ax.plot(t_axis, pred_actions[b, :, d].cpu(), label="pred", lw=2, ls="--")
        if b == 0:
            ax.set_title(names[d])
        if d == 0:
            ax.set_ylabel(f"traj {b}")
        ax.axhline(0, color="gray", lw=0.5)
    axes[b, 0].legend(loc="upper right", fontsize=8)
plt.suptitle("Inverse-dynamics probe: predicted vs target actions")
plt.tight_layout()
plt.show()


# ============================================================
# Decoder probe: z (predicted) -> image, compare to target.
# Frozen encoder. Decoder trained on (z_target, target_image) pairs.
# ============================================================

device = next(model.parameters()).device
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

img_hw = 28
z_dim  = None  # discovered from a forward pass below

# -- 1. Decoder: z -> 28x28 image --
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden=512, img_hw=28):
        super().__init__()
        self.img_hw = img_hw
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, img_hw * img_hw),
            nn.Sigmoid(),
        )
    def forward(self, z):  # z: (..., z_dim)
        x = self.net(z)
        return x.view(*z.shape[:-1], 1, self.img_hw, self.img_hw)

# discover z_dim
with torch.no_grad():
    imgs, _ = next(iter(train_loader))
    imgs = imgs.squeeze(1).to(device)
    _, _, input_images, _ = rollout(imgs, T_max, scale_sensitivity, translation_sensitivity, device=device)
    z_probe, _ = model.encode(input_images)
    z_dim = z_probe.size(-1)

decoder = Decoder(z_dim, hidden=512, img_hw=img_hw).to(device)
opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)

# -- 2. Train decoder on frozen encoder outputs --
# We map z_target -> target_image so the decoder learns the encoder's inverse.
n_steps = 10000
for step in range(n_steps):
    imgs, _ = next(iter(train_loader))
    imgs = imgs.squeeze(1).to(device)
    _, _, input_images, target_images = rollout(
        imgs, T_max, scale_sensitivity, translation_sensitivity, device=device,
    )
    with torch.no_grad():
        # z_preds, z_img, z_action = model(input_images, actions)
        z_img, _ = model.encode(target_images)    

    recon = decoder(z_img)                        # (B, T, 1, H, W)
    loss = F.mse_loss(recon, target_images)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 200 == 0:
        print(f"step {step:4d}  decoder MSE {loss.item():.4f}")

# -- 3. Eval: decode the PREDICTOR's outputs, compare to target images --
decoder.eval()
with torch.no_grad():
    imgs, _ = next(iter(val_loader))
    imgs = imgs.squeeze(1).to(device)
    _, actions, input_images, target_images = rollout(
        imgs, T_max, scale_sensitivity, translation_sensitivity, device=device,
    )
    z_pred, _, _ = model(input_images, actions, ar_steps=T_max)         # (B, T, D)  predictor output
    img_pred = decoder(z_pred)                     # (B, T, 1, H, W)

# -- 4. Plot: rows alternate target / predicted, columns = time --
n_traj = 6
fig, axes = plt.subplots(2 * n_traj, T_max, figsize=(T_max * 1.0, 2 * n_traj * 1.0))
for b in range(n_traj):
    for t in range(T_max):
        axes[2*b,     t].imshow(target_images[b, t, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[2*b + 1, t].imshow(img_pred[b, t, 0].cpu(),       cmap="gray", vmin=0, vmax=1)
        for r in (2*b, 2*b + 1):
            axes[r, t].set_xticks([])
            axes[r, t].set_yticks([])
    axes[2*b,     0].set_ylabel(f"traj {b}\ntarget", fontsize=8)
    axes[2*b + 1, 0].set_ylabel("decoded\npred",     fontsize=8)
plt.suptitle("Target glimpse  vs  decode(predictor(z_t, a_t))")
plt.tight_layout()
plt.show()




#================================================
#                SURPRISE EVAL
#================================================
# %%
@torch.no_grad()
def measure_surprise(model, input_images, actions, target_images):
    """Per-step MSE between predicted and target embeddings.
    Returns shape (B, T)."""
    z_pred, _, _ = model(input_images, actions, ar_steps=T_max)
    z_target, _ = model.encode(target_images)
    return ((z_pred - z_target) ** 2).mean(dim=-1)   # (B, T)

@torch.no_grad()
def rollout_with_perturbation(imgs, T_max, t_perturb, mode, device,
                              scale_sensitivity=0.2, translation_sensitivity=0.1):
    """
    mode: "none" | "teleport" | "swap_digit" | "invert"
      - none:        clean baseline
      - teleport:    physical violation — at t_perturb, glimpse state jumps to random
      - swap_digit:  identity violation — at t_perturb, underlying MNIST image swaps
      - invert:      visual perturbation — at t_perturb, intensities are inverted (1 - x)
    """
    seed, actions, input_images, target_images = rollout(
        imgs, T_max, scale_sensitivity, translation_sensitivity, device=device,
    )
    if mode == "none":
        return actions, input_images, target_images

    B = imgs.shape[0]

    if mode == "teleport":
        # Replace the action at t_perturb with a large random jump,
        # so the actual observed next frame is far from what the action implies.
        # Easiest: re-render the target frame at a completely random state.
        from glimpse import Glimpse
        random_state = torch.randn(B, 3, device=device) * 1.0  # log_scale, x, y
        g = Glimpse(imgs,
                    log_scale=random_state[:, 0:1],
                    x=random_state[:, 1:2],
                    y=random_state[:, 2:3],
                    T_max=1)
        teleported = g.transform(t=-1)                          # (B, H, W)
        target_images[:, t_perturb, 0] = teleported

    elif mode == "swap_digit":
        # Replace underlying digits with a different batch from t_perturb onward.
        other_imgs, _ = next(iter(train_loader))
        other_imgs = other_imgs.squeeze(1)[:B].to(device)
        _, _, _, other_targets = rollout(
            other_imgs, T_max, scale_sensitivity, translation_sensitivity, device=device,
        )
        target_images[:, t_perturb:] = other_targets[:, t_perturb:]

    elif mode == "invert":
        # Visual-only perturbation (no spatial change)
        target_images[:, t_perturb:] = 1.0 - target_images[:, t_perturb:]

    return actions, input_images, target_images

import matplotlib.pyplot as plt

device = next(model.parameters()).device
model.eval()
t_perturb = T_max // 2   # perturb at the middle of the trajectory

modes = ["none", "teleport", "swap_digit", "invert"]
labels = ["unperturbed", "teleport", "swap digit", "invert"]
colors = ["gray", "tab:red", "tab:orange", "tab:blue"]

# Collect surprise curves over several batches per condition
curves = {m: [] for m in modes}
n_batches = 10
for _ in range(n_batches):
    imgs, _ = next(iter(val_loader))
    imgs = imgs.squeeze(1).to(device)
    for m in modes:
        actions, input_images, target_images = rollout_with_perturbation(
            imgs, T_max, t_perturb, mode=m, device=device,
        )
        s = measure_surprise(model, input_images, actions, target_images)  # (B, T)
        curves[m].append(s.cpu())

# stack into (N, T) per condition
curves = {m: torch.cat(curves[m], dim=0) for m in modes}

# Plot
plt.figure(figsize=(8, 4))
for m, lbl, c in zip(modes, labels, colors):
    s = curves[m]
    mean = s.mean(0)
    std  = s.std(0)
    t = torch.arange(s.size(-1))
    plt.plot(t, mean, label=lbl, color=c, lw=2)
    plt.fill_between(t, mean - std, mean + std, color=c, alpha=0.15)
plt.axvline(t_perturb, color="k", ls="--", lw=0.7, label="perturbation")
plt.xlabel("step t")
plt.ylabel("surprise (MSE in latent space)")
plt.title("Violation-of-expectation: per-step prediction error")
plt.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, len(modes), figsize=(4 * len(modes), 4),
                         sharey=True, sharex=True)

for ax, m, lbl, c in zip(axes, modes, labels, colors):
    s = curves[m]               # (N, T)
    t = torch.arange(s.size(-1))

    # individual trajectories — thin, semi-transparent
    for i in range(min(s.size(0), 30)):     # cap at 30 lines so plot stays readable
        ax.plot(t, s[i], color=c, alpha=0.15, lw=0.8)

    # mean — bold
    ax.plot(t, s.mean(0), color=c, lw=2.5, label=f"{lbl} (mean)")

    ax.axvline(t_perturb, color="k", ls="--", lw=0.7)
    ax.set_title(lbl)
    ax.set_xlabel("step t")
    ax.legend(loc="upper left", fontsize=8)

axes[0].set_ylabel("surprise (MSE)")
plt.suptitle("Per-trajectory surprise curves")
plt.tight_layout()
plt.show()


#------------------------------------------------
#          plot trajectories and mode graphs
#------------------------------------------------
# %%
from matplotlib.gridspec import GridSpec

def show_examples_with_decode(mode, decoder=None, n_examples=3, color="tab:red"):
    """Per traj: target row above pred row (frames flush); surprise curve on right."""
    if decoder is None:
        decoder = getattr(model, "decoder", None)
    assert decoder is not None, "No decoder available."

    imgs, _ = next(iter(val_loader))
    imgs = imgs.squeeze(1).to(device)

    actions, input_images, target_images = rollout_with_perturbation(
        imgs, T_max, t_perturb, mode=mode, device=device,
    )
    with torch.no_grad():
        z_pred, _, _ = model(input_images, actions, ar_steps=T_max)
        img_pred = decoder(z_pred)
        z_target, _ = model.encode(target_images)
        surprise = ((z_pred - z_target) ** 2).mean(dim=-1).cpu()

    n_rows = 2 * n_examples
    fig = plt.figure(figsize=(T_max * 0.8 + 4, n_rows * 0.9))

    # Outer split: frames on the left, curves on the right
    outer = GridSpec(1, 2, figure=fig, width_ratios=[T_max, 5], wspace=0.15)

    # Frames sub-grid: zero spacing so pixels touch
    frame_gs = outer[0].subgridspec(n_rows, T_max, wspace=0, hspace=0)

    # Curve sub-grid: one curve per trajectory (n_examples rows)
    curve_gs = outer[1].subgridspec(n_examples, 1, hspace=0.4)

    for b in range(n_examples):
        r_tgt, r_prd = 2 * b, 2 * b + 1

        # --- target row ---
        for t in range(T_max):
            ax = fig.add_subplot(frame_gs[r_tgt, t])
            ax.imshow(target_images[b, t, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(f"traj{b}\ntarget", fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=20)
            if t == t_perturb:
                for s in ax.spines.values():
                    s.set_color("red"); s.set_linewidth(2)

        # --- pred row ---
        for t in range(T_max):
            ax = fig.add_subplot(frame_gs[r_prd, t])
            ax.imshow(img_pred[b, t, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(f"traj{b}\npred", fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=20)
            if t == t_perturb:
                for s in ax.spines.values():
                    s.set_color("red"); s.set_linewidth(2)

        # --- surprise curve, one per trajectory in the right column ---
        ax_curve = fig.add_subplot(curve_gs[b, 0])
        ax_curve.plot(range(T_max), surprise[b], color=color, lw=2)
        ax_curve.axvline(t_perturb, color="k", ls="--", lw=0.7)
        ax_curve.set_xlabel("step t", fontsize=8)
        ax_curve.set_ylabel("surprise", fontsize=8)
        ax_curve.set_ylim(0, surprise.max().item() * 1.1)
        ax_curve.tick_params(labelsize=7)

    plt.suptitle(f"Mode: {mode}", y=1.0, fontsize=11)
    plt.show()


show_examples_with_decode("teleport",   n_examples=3, color="tab:red")
show_examples_with_decode("swap_digit", n_examples=3, color="tab:orange")
show_examples_with_decode("invert",     n_examples=3, color="tab:blue")

# %%
