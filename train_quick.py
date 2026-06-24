"""
A script to quickly train an ML model and test things quickly without config. 
Keeps it flexible for experimenting fast
"""

import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from einops import rearrange
import matplotlib.pyplot as plt
import wandb

from glimpse import rollout
from jepa import JEPA
from ml_layers import ARPredictor, ActionEncoder, ImageEncoder, Decoder, MLP_Projector, MemoryPredictor
from vis_utils import plot_glimpse_frames


#================================================
#                GLOBAL VARS
#================================================
scale_sensitivity = 0.2
translation_sensitivity = 0.1
batch_size = 64
T_max = 10
lr = 4e-4
epochs = 20 #20
viz_every = 1
run_suffix = "noMemoryDMT-win3-mse" # optional tag appended to the dated plot dir: out/plots/YY_MMDD-{run_suffix}/
lambd = 0.09 # sigreg loss coefficient
lambd_recon = 0.1
ar_steps = 0 # Teacher-forcing (used when ar_curriculum is False)

# NextLat-style objective (arXiv:2511.05963)
latent_loss = "mse" # "mse" (baseline) or "smooth_l1" (NextLat robustness)
smooth_l1_beta = 1.0

# DMT post-finetune (DAgger Memory Training): after training, freeze encoder+decoder and
# unroll the predictor on its own predictions, regressing to the frozen encoder trajectory.
# Corrects AR drift. Flip run_dmt=False to disable (behavior then unchanged).
run_dmt = True
dmt_steps = 500
dmt_lr = 1e-4

# model params
input_dim_action = 3 # log_scale, x, y are only 3 parameters
hidden_dim_img_encoder = 512
hidden_dim_predictor = 512

decoder_hidden_dim = 512
z_dim_img = 36
z_dim_action = 3
depth_img_encoder = 3
depth_predictor = 2
context_window = 3 # bound predictor attention to last N frames; None = full causal prefix

# memory predictor params
use_memory = False
# "content" = memory concatenated to latent vector z
# "adaln" = memory concatenated onto the action cond.
mem_mode = "content"
z_dim_memory = 36
mem_hidden_dim = 256
mem_depth = 2
mem_heads = 4


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

# predictor = ARPredictorSimpleAdaLN(z_dim_img, z_dim_action, hidden_dim_predictor)
# predictor = ARPredictorSimple(z_dim_img, z_dim_action, hidden_dim_predictor, depth=depth_predictor)
# cond width = action (+ memory only in "adaln"); "content" mode concatenates memory onto the INPUT.
cond_dim = z_dim_action + (z_dim_memory if (use_memory and mem_mode == "adaln") else 0)
# input WIDTH the predictor must accept: in "content" mode predict() does torch.cat([z_img, memory]),
# so the input vector is z_dim_img + z_dim_memory wide (e.g. 36 + 36 = 72). This is dim sizing, not a sum.
pred_input_dim = z_dim_img + (z_dim_memory if (use_memory and mem_mode == "content") else 0)
predictor = ARPredictor(num_frames=T_max, depth=4, heads=4, mlp_dim=512, input_dim=pred_input_dim, hidden_dim=hidden_dim_predictor, output_dim=hidden_dim_predictor, action_dim=cond_dim, window=context_window)
projector_pred = MLP_Projector(input_dim=hidden_dim_predictor, output_dim=z_dim_img, hidden_dim=256, norm_fn=torch.nn.BatchNorm1d)
decoder = Decoder(z_dim=z_dim_img, hidden_dim=decoder_hidden_dim, h=H, w=W, depth=2)
memory_predictor = MemoryPredictor(z_dim_img, z_dim_memory, hidden_dim=mem_hidden_dim, depth=mem_depth, heads=mem_heads) if use_memory else None
# Pin the encoder-output scale. SigReg alone fails to control latent scale here, letting it
# drift (z_std 2-25) and destabilize training/decode; a non-affine LayerNorm fixes it.
projector = nn.LayerNorm(z_dim_img, elementwise_affine=False)

model = JEPA(image_encoder, predictor, action_encoder, decoder=decoder, projector=projector, projector_pred=projector_pred, memory_predictor=memory_predictor, mem_mode=mem_mode)

# 2. Move the model to the right device (cuda if available, else cpu).
model = model.to(device)

# 3. Define an Adam optimizer over the model parameters with learning rate lr.
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# 4. Define an MSE loss to compare predicted glimpse against the true next glimpse.
loss_fn = nn.MSELoss()


#================================================
#                WANDB INIT
#================================================
wandb.init(
    project="glimpse-jepa",
    config={
        "scale_sensitivity": scale_sensitivity,
        "translation_sensitivity": translation_sensitivity,
        "batch_size": batch_size,
        "T_max": T_max,
        "lr": lr,
        "epochs": epochs,
        "lambd_sigreg": lambd,
        "lambd_recon": lambd_recon,
        "input_dim_action": input_dim_action,
        "hidden_dim_img_encoder": hidden_dim_img_encoder,
        "hidden_dim_predictor": hidden_dim_predictor,
        "decoder_hidden_dim": decoder_hidden_dim,
        "z_dim_img": z_dim_img,
        "z_dim_action": z_dim_action,
        "z_dim_memory": z_dim_memory,
        "depth_img_encoder": depth_img_encoder,
        "depth_predictor": depth_predictor,
    },
)
global_step = 0

# OOD rollout length: trained on T_max, tested at 2x to probe generalization
T_ood = 2 * T_max

# dated output dir for saved plots: out/plots/YY_MMDD[-run_suffix]/
date_tag = datetime.now().strftime("%y_%m%d")
plot_dir = os.path.join("out", "plots", f"{date_tag}-{run_suffix}" if run_suffix else date_tag)
os.makedirs(plot_dir, exist_ok=True)

# history for end-of-run plots
hist = {"step": [], "train_mse": [],
        "epoch": [], "train_loss": [], "val_loss": [], "val_mse_tf": [], "val_mse_ar": [],
        "val_recon_enc": [], "val_recon_pred": []}


def plot_target_vs_decode(target, decode, n=4, title=""):
    """Rows alternate target / decoded-prediction; columns = rollout steps."""
    Tn = target.size(1)
    fig, axes = plt.subplots(2 * n, Tn, figsize=(Tn * 0.6, 2 * n * 0.6))
    for b in range(n):
        for t in range(Tn):
            axes[2 * b, t].imshow(target[b, t, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            axes[2 * b + 1, t].imshow(decode[b, t, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            for r in (2 * b, 2 * b + 1):
                axes[r, t].set_xticks([]); axes[r, t].set_yticks([])
            if t == T_max - 1 and Tn > T_max: # mark train horizon boundary
                for r in (2 * b, 2 * b + 1):
                    for sp in axes[r, t].spines.values():
                        sp.set_color("red"); sp.set_linewidth(1.5)
        axes[2 * b, 0].set_ylabel(f"t{b}\ntgt", fontsize=7)
        axes[2 * b + 1, 0].set_ylabel("pred", fontsize=7)
    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    return fig


#================================================
#                TRAIN LOOP
#================================================
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
            
        # 6. Inference (teacher-forcing, therefore ar_steps are 0)
        z_preds, z_img, z_action = model(input_images, actions, ar_steps=0)

        # # 7. Simple MLP Loss. MSE over all T_max predicted frames vs true frames
        # loss = loss_fn(preds, target_images)
        
        # 7. JEPA loss
        # encode target images
        z_targets, _ = model.encode(target_images)
        # get loss between predicted latent vector and actual latent vector
        loss_mse = model.mse(z_preds, z_targets, mean=False, loss_type=latent_loss, beta=smooth_l1_beta)
        # get sigreg loss
        loss_sigreg = model.sigreg_loss(z_img)
        # recon loss on encoder's latents
        loss_recon  = model.recon_loss(z_preds.detach(), target_images) # use this if you want recon loss to effect predictor
        # loss_recon = model.recon_loss(z_img.detach(), input_images) # Use this if you don't want recon loss to effect predictor or encoder weights
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
        hist["step"].append(global_step)
        hist["train_mse"].append(loss_mse.item())

        # TODO: add to log a config file
        wandb.log({
            "train/loss": loss.item(),
            "train/loss_mse": loss_mse.item(),
            "train/loss_sigreg": loss_sigreg.item(),
            "train/loss_recon": loss_recon.item(),
            "train/ar_steps": 0,
            "epoch": epoch,
        }, step=global_step)
        global_step += 1


    # ---- validation ----
    model.eval() # set mode to eval mode
    
    # initiate loss value
    val_batches = 0    
    val_loss_sum = 0.0
    val_mse_tf_sum = 0.0
    val_mse_ar_sum = 0.0
    val_recon_enc_sum = 0.0
    val_recon_pred_sum = 0.0
    val_pbar = tqdm(val_loader, desc=f"val {epoch}", leave=False)
    
    with torch.no_grad():
        for imgs, _ in val_pbar:
            
            # 1. Get dataset rollout with random actions
            imgs = imgs.to(device).squeeze(1)
            seed, actions, input_images, target_images = rollout(imgs, T_max, scale_sensitivity, translation_sensitivity, device=device)
            z_targets, _ = model.encode(target_images)

            # Autoregressive latent MSE (full rollout, no teacher-forcing) — the long-horizon metric
            z_preds, _, _ = model(input_images, actions, ar_steps=input_images.size(1))
            mse_ar = model.mse(z_preds, z_targets, mean=False)
            
            # Teacher-forced latent MSE (one-step prediction quality)
            # This is to compare it to auto-regressive output
            z_preds_tf, z_img, _ = model(input_images, actions, ar_steps=0)
            mse_tf = model.mse(z_preds_tf, z_targets, mean=False)
                        
            # Decoder recon quality: encoder latents vs inputs, predicted latents vs targets
            recon_enc = model.recon_loss(z_img, input_images)
            recon_pred = model.recon_loss(z_preds, target_images)

            # Compute and append loss
            loss_sigreg = model.sigreg_loss(z_img)
            loss = mse_ar + lambd * loss_sigreg
            val_loss_sum += loss.item()
            val_mse_tf_sum += mse_tf.item()
            val_mse_ar_sum += mse_ar.item()
            val_recon_enc_sum += recon_enc.item()
            val_recon_pred_sum += recon_pred.item()
            val_batches  += 1

    train_avg = train_loss_sum / max(train_batches, 1)
    val_avg   = val_loss_sum   / max(val_batches, 1)
    val_mse_tf = val_mse_tf_sum / max(val_batches, 1)
    val_mse_ar = val_mse_ar_sum / max(val_batches, 1)
    val_recon_enc = val_recon_enc_sum / max(val_batches, 1)
    val_recon_pred = val_recon_pred_sum / max(val_batches, 1)
    print(f"epoch {epoch:4d}  train {train_avg:.4f}  val {val_avg:.4f}  "
          f"val_mse_tf {val_mse_tf:.4f}  val_mse_ar {val_mse_ar:.4f}  "
          f"recon_enc {val_recon_enc:.4f}  recon_pred {val_recon_pred:.4f}")

    hist["epoch"].append(epoch)
    hist["train_loss"].append(train_avg)
    hist["val_loss"].append(val_avg)
    hist["val_mse_tf"].append(val_mse_tf)
    hist["val_mse_ar"].append(val_mse_ar)
    hist["val_recon_enc"].append(val_recon_enc)
    hist["val_recon_pred"].append(val_recon_pred)

    epoch_log = {
        "epoch": epoch,
        "train/avg_loss": train_avg,
        "val/avg_loss": val_avg,
        "val/mse_tf": val_mse_tf,
        "val/mse_ar": val_mse_ar,
        "val/recon_enc": val_recon_enc,
        "val/recon_pred": val_recon_pred,
    }

    # Every N epochs, render true vs predicted glimpse sequences from the last val batch
    if epoch % viz_every == 0:
        # get last target images from validation
        true_seq = target_images.squeeze(2).cpu() # (B, T_max, 28, 28)
        
        # get last z_preds from validation
        s = int(z_dim_img**0.5)
        z_preds_img = rearrange(z_preds, "b t (h w) -> b t h w", h=s, w=s) # Reshape latent vector to an image (just for viz)
        pred_seq = z_preds_img.squeeze(2).detach().cpu() # (B, T_max, 28, 28)

        # Reconstruct image from predicted latent vectors and from encoder latent vectors
        recon_images_from_predictor = model.decode(z_preds).squeeze(2).detach().cpu()
        recon_images_from_encoder = model.decode(z_img).squeeze(2).detach().cpu()

        fig_true        = plot_glimpse_frames(true_seq, title="True target frames")
        fig_recon_enc   = plot_glimpse_frames(recon_images_from_encoder, title="Decoded from encoder latents")
        fig_recon_pred  = plot_glimpse_frames(recon_images_from_predictor, title="Decoded from predicted latents")
        fig_pred_latent = plot_glimpse_frames(pred_seq, title="Predicted latents (reshaped to image)")

        epoch_log.update({
            "viz/true":             wandb.Image(fig_true),
            "viz/recon_encoder":    wandb.Image(fig_recon_enc),
            "viz/recon_predictor": wandb.Image(fig_recon_pred),
            "viz/pred_latent":      wandb.Image(fig_pred_latent),
        })
        plt.close(fig_true)
        plt.close(fig_recon_enc)
        plt.close(fig_recon_pred)
        plt.close(fig_pred_latent)

        # OOD rollout: trained on T_max, roll out to T_ood (2x) to test generalization
        with torch.no_grad():
            _, actions_o, input_o, target_o = rollout(imgs, T_ood, scale_sensitivity, translation_sensitivity, device=device)
            z_pred_o, _, _ = model(input_o, actions_o, ar_steps=input_o.size(1)) # full AR
            decode_o = model.decode(z_pred_o)
        fig_ood = plot_target_vs_decode(target_o, decode_o, n=4, title=f"OOD rollout T={T_ood} (red = train horizon T={T_max})")
        epoch_log["viz/ood_rollout"] = wandb.Image(fig_ood)
        plt.close(fig_ood)

        # Collapse detector
        with torch.no_grad():
            z_flat = z_img.reshape(-1, z_img.size(-1)) # (N, D)
            z_std  = z_flat.std(0).mean().item() # ~0 = collapsed, ~1 = healthy
            z_norm = z_flat.norm(dim=-1).mean().item()
            # pairwise cosine of random pairs — should not be ~1
            a, b = z_flat[:50], z_flat[50:100]
            cos = F.cosine_similarity(a, b).mean().item()
        print(f"std={z_std:.3f}  norm={z_norm:.3f}  cos={cos:.3f}")
        
        epoch_log.update({
            "collapse/z_std": z_std,
            "collapse/z_norm": z_norm,
            "collapse/cos_sim": cos,
        })

    wandb.log(epoch_log, step=global_step)


#================================================
#            DMT POST-FINETUNE
#================================================
# DAgger Memory Training: freeze encoder+decoder, unroll the predictor on its own
# predictions and regress to the frozen encoder trajectory. Cuts AR drift.

@torch.no_grad()
def eval_val_ar(m):
    """Mean val AR latent nMSE over the full val set (full autoregressive rollout)."""
    m.eval()
    tot, n = 0.0, 0
    for imgs, _ in val_loader:
        imgs = imgs.to(device).squeeze(1)
        _, acts, inp, tgt = rollout(imgs, T_max, scale_sensitivity, translation_sensitivity, device=device)
        z_tgt, _ = m.encode(tgt)
        z_pred, _, _ = m(inp, acts, ar_steps=inp.size(1))
        tot += m.mse(z_pred, z_tgt, mean=False).item(); n += 1
    return tot / max(n, 1)


@torch.no_grad()
def ar_decode_and_perstep(m, inp, acts, tgt):
    """Full AR rollout -> decoded frames (B, T, 1, H, W) and per-step pixel MSE (T,)."""
    m.eval()
    z_pred, _, _ = m(inp, acts, ar_steps=inp.size(1))
    decode = m.decode(z_pred)
    perstep = ((decode - tgt.float()) ** 2).mean(dim=(0, 2, 3, 4))  # avg over batch + pixels
    return decode, perstep


def plot_dmt_decode(target, dec_before, dec_after, n=4, title=""):
    """Per trajectory: target / decode-before / decode-after rows; columns = rollout steps."""
    Tn = target.size(1)
    fig, axes = plt.subplots(3 * n, Tn, figsize=(Tn * 0.6, 3 * n * 0.6))
    rows = [("tgt", target), ("before", dec_before), ("after", dec_after)]
    for b in range(n):
        for k, (lbl, src) in enumerate(rows):
            r = 3 * b + k
            for t in range(Tn):
                axes[r, t].imshow(src[b, t, 0].cpu(), cmap="gray", vmin=0, vmax=1)
                axes[r, t].set_xticks([]); axes[r, t].set_yticks([])
                if t == T_max - 1 and Tn > T_max: # mark train horizon boundary
                    for sp in axes[r, t].spines.values():
                        sp.set_color("red"); sp.set_linewidth(1.5)
            axes[r, 0].set_ylabel(f"t{b}\n{lbl}", fontsize=7)
    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    return fig


if run_dmt:
    print("\n=== DMT post-finetune ===")

    # Fixed viz batch so before/after compare the SAME target glimpses (rolled to T_ood for drift)
    viz_imgs, _ = next(iter(val_loader))
    viz_imgs = viz_imgs.to(device).squeeze(1)
    with torch.no_grad():
        _, viz_actions, viz_inp, viz_tgt = rollout(viz_imgs, T_ood, scale_sensitivity, translation_sensitivity, device=device)

    ar_before = eval_val_ar(model)
    dec_before, perstep_before = ar_decode_and_perstep(model, viz_inp, viz_actions, viz_tgt)

    # Freeze all but the predictor; keep frozen BatchNorm (projector_pred) stats fixed via eval()
    params = model.freeze_for_dmt()
    dmt_opt = torch.optim.AdamW(params, lr=dmt_lr)
    model.eval(); model.predictor.train()

    dmt_it = iter(train_loader)
    dmt_pbar = tqdm(range(dmt_steps), desc="dmt")
    for step in dmt_pbar:
        try: imgs, _ = next(dmt_it)
        except StopIteration: dmt_it = iter(train_loader); imgs, _ = next(dmt_it)
        imgs = imgs.to(device).squeeze(1)
        with torch.no_grad():
            _, acts, inp, tgt = rollout(imgs, T_max, scale_sensitivity, translation_sensitivity, device=device)
        loss = model.dmt_loss(inp, acts, tgt)
        dmt_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        dmt_opt.step()
        dmt_pbar.set_postfix(loss=f"{loss.item():.4f}")
        wandb.log({"dmt/loss": loss.item()}, step=global_step); global_step += 1

    ar_after = eval_val_ar(model)
    dec_after, perstep_after = ar_decode_and_perstep(model, viz_inp, viz_actions, viz_tgt)
    print(f"DMT: val AR latent nMSE {ar_before:.4f} -> {ar_after:.4f}")

    # Plot 1: target vs decode(AR predictor) before/after, same glimpses
    viz_tgt_img = viz_tgt.float()
    fig_dmt_dec = plot_dmt_decode(viz_tgt_img, dec_before, dec_after, n=4,
                                  title=f"DMT decode AR (T={T_ood}, red=train horizon T={T_max})")
    fig_dmt_dec.savefig(os.path.join(plot_dir, "dmt_decode_before_after.png"), dpi=110)

    # Plot 2: per-step decoder recon loss before/after
    fig_dmt_ps, axp = plt.subplots(figsize=(8, 4))
    steps_axis = range(1, T_ood + 1)
    axp.plot(steps_axis, perstep_before.cpu(), marker="o", label="before DMT")
    axp.plot(steps_axis, perstep_after.cpu(), marker="o", label="after DMT")
    axp.axvline(T_max, color="k", ls="--", lw=0.7, label="train horizon")
    axp.set_xlabel("rollout step"); axp.set_ylabel("decoder recon MSE")
    axp.set_yscale("log"); axp.set_title("Per-step AR decoder recon: before vs after DMT")
    axp.legend(fontsize=8); axp.grid(alpha=0.3)
    plt.tight_layout()
    fig_dmt_ps.savefig(os.path.join(plot_dir, "dmt_perstep_recon.png"), dpi=110)

    wandb.log({
        "dmt/ar_before": ar_before,
        "dmt/ar_after": ar_after,
        "dmt/decode_before_after": wandb.Image(fig_dmt_dec),
        "dmt/perstep_recon": wandb.Image(fig_dmt_ps),
    }, step=global_step)
    plt.close(fig_dmt_dec); plt.close(fig_dmt_ps)

    # Re-enable grads so downstream probes behave normally
    for p in model.parameters():
        p.requires_grad_(True)


#================================================
#            TRAINING HISTORY PLOTS
#================================================
# 1. Training history: per-step train latent MSE
fig_hist, ax = plt.subplots(figsize=(8, 4))
ax.plot(hist["step"], hist["train_mse"], lw=0.8, alpha=0.8)
ax.set_xlabel("step")
ax.set_ylabel("train latent MSE")
ax.set_yscale("log")
ax.set_title("Training history: per-step latent MSE")
ax.grid(alpha=0.3)
plt.tight_layout()
fig_hist.savefig(os.path.join(plot_dir, "train_history.png"), dpi=110)

# 2. Latent MSE over epochs: teacher-forced vs autoregressive (val)
fig_mse, ax = plt.subplots(figsize=(8, 4))
ax.plot(hist["epoch"], hist["val_mse_tf"], marker="o", label="val MSE (teacher-forced)")
ax.plot(hist["epoch"], hist["val_mse_ar"], marker="o", label="val MSE (autoregressive)")
ax.plot(hist["epoch"], hist["train_loss"], marker=".", ls="--", alpha=0.6, label="train total loss")
ax.set_xlabel("epoch")
ax.set_ylabel("latent MSE / loss")
ax.set_yscale("log")
ax.set_title("Latent MSE over training")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
fig_mse.savefig(os.path.join(plot_dir, "latent_mse_history.png"), dpi=110)

# 3. Decoder recon MSE over epochs: encoder latents vs predicted latents (val)
fig_recon, ax = plt.subplots(figsize=(8, 4))
ax.plot(hist["epoch"], hist["val_recon_enc"], marker="o", label="recon (encoder latents)")
ax.plot(hist["epoch"], hist["val_recon_pred"], marker="o", label="recon (predicted latents)")
ax.set_xlabel("epoch")
ax.set_ylabel("pixel MSE")
ax.set_yscale("log")
ax.set_title("Decoder reconstruction over training")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
fig_recon.savefig(os.path.join(plot_dir, "recon_history.png"), dpi=110)

wandb.log({"history/train_mse": wandb.Image(fig_hist),
           "history/latent_mse": wandb.Image(fig_mse),
           "history/recon": wandb.Image(fig_recon)}, step=global_step)
plt.close(fig_hist); plt.close(fig_mse); plt.close(fig_recon)


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
n_steps = 5000
for step in range(n_steps):
    imgs, _ = next(iter(train_loader))
    imgs = imgs.squeeze(1).to(device)
    _, _, input_images, target_images = rollout(
        imgs, T_max, scale_sensitivity, translation_sensitivity, device=device,
    )
    with torch.no_grad():
        # z_preds, z_img, z_action = model(input_images, actions, ar_steps=input_images.size(1))
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
    z_pred, _, _ = model(input_images, actions, ar_steps=input_images.size(1))         # (B, T, D)  predictor output
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
@torch.no_grad()
def measure_surprise(model, input_images, actions, target_images):
    """Per-step MSE between predicted and target embeddings.
    Returns shape (B, T)."""
    z_pred, _, _ = model(input_images, actions, ar_steps=input_images.size(1))
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
        z_pred, _, _ = model(input_images, actions, ar_steps=input_images.size(1))
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


