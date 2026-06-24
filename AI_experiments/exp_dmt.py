"""Two attacks on AR rollout drift, judged on AR latent nMSE (the metric we care about):

Exp 1 (latent size): spatial JEPA at C in {2,4,8} -> 32/64/128-float latent (C=2 ~ baseline 36).
                     Does a smaller spatial latent roll out better?
Exp 2 (DMT):         DAgger Memory Training post-finetune (Maes et al. / "Pretraining Recurrent
                     Networks without Recurrence", arXiv:2606.06479). Freeze encoder+decoder, unroll
                     the predictor on its OWN predicted latents, regress each step to the FROZEN
                     encoder trajectory: L = E_t[MSE(z_hat_t, z_t)]. Small LR, post-training.
                     Applied to flat baseline + spatial; AR measured before vs after.

Note: their RNN-DMT teacher-forces the observation x_t each step; our world model has no future
observations at rollout (seed + actions only), so this is the closed-loop adaptation of DMT.

Run: "C:/Users/Ous/miniconda3/envs/ML/python.exe" exp_dmt.py [pretrain_steps] [dmt_steps]
"""
import sys, json
import torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import exp_ideas as E
from exp_spatial_jepa import SpatialJEPA, OUT, plot_rollout
from glimpse import rollout

DEV, T = E.DEV, E.T


# ---- on-policy AR unroll WITH gradient (feeds the model its own predictions) ----
def spatial_ar_grad(m, inp, actions):
    with torch.no_grad():
        z_seed = m.encode(inp)[:, :1]
    z_seq = z_seed
    for t in range(T):
        pred = m.pred(z_seq, actions[:, :t + 1])[:, -1:]
        z_seq = torch.cat([z_seq, pred], dim=1)
    return z_seq[:, 1:]


def mem_ar_grad(m, inp, actions):
    j = m.jepa
    with torch.no_grad():
        z_img, _ = j.encode(inp)          # frozen encoder
    z_act = j.action_encoder(actions)     # trainable conditioning
    z_in = z_img[:, :1]
    for t in range(T):
        mem = j.predict_memory(z_in) if j.memory_predictor is not None else None
        if m.mode == "adaln":
            cond = z_act[:, :t + 1] if mem is None else torch.cat([z_act[:, :t + 1], mem], dim=-1)
            raw = j.predictor(z_in, cond)[:, -1:]
        else:
            x = z_in + (j.mem_proj(mem) if mem is not None else 0)
            raw = j.predictor(x, z_act[:, :t + 1])[:, -1:]
        z_in = torch.cat([z_in, j.project(raw)], dim=1)
    return z_in[:, 1:]


def encode_tgt(m, kind, tgt):
    with torch.no_grad():
        return m.encode(tgt) if kind == "spatial" else m.jepa.encode(tgt)[0]


def frozen(kind, name):
    if kind == "spatial":
        return name.startswith(("enc.", "dec."))
    return name.startswith(("jepa.encoder", "jepa.decoder"))  # freeze encoder + decoder only


def dmt(model, kind, steps=500, lr=1e-4):
    for n, p in model.named_parameters():
        p.requires_grad_(not frozen(kind, n))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    loader = E.make_train_loader(); it = iter(loader)
    model.train()
    for step in range(steps):
        try: imgs, _ = next(it)
        except StopIteration: it = iter(loader); imgs, _ = next(it)
        imgs = imgs.to(DEV).squeeze(1)
        with torch.no_grad():
            _, actions, inp, tgt = rollout(imgs, T, E.SCALE_S, E.TRANS_S, device=DEV)
        z_tgt = encode_tgt(model, kind, tgt)
        z_ar = spatial_ar_grad(model, inp, actions) if kind == "spatial" else mem_ar_grad(model, inp, actions)
        loss = F.mse_loss(z_ar, z_tgt.detach())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
    return loss.item()


@torch.no_grad()
def eval_ar(model, vb):
    model.eval()
    _, actions, inp, tgt = rollout(vb, T, E.SCALE_S, E.TRANS_S, device=DEV)
    return model.eval_ar(inp, actions, tgt)  # (met, pack)


def main(pre=1500, dmt_steps=500):
    vb = E.val_batch()
    configs = [
        ("flat_baseline", lambda: E.MemModel("adaln", "flat_baseline"), "mem"),
        ("spatial_c2", lambda: SpatialJEPA(c=2, name="spatial_c2"), "spatial"),   # ~32f (~baseline 36)
        ("spatial_c4", lambda: SpatialJEPA(c=4, name="spatial_c4"), "spatial"),   # 64f
        ("spatial_c8", lambda: SpatialJEPA(c=8, name="spatial_c8"), "spatial"),   # 128f
    ]
    res = {}
    for name, build, kind in configs:
        m = build().to(DEV)
        E.train_model(m, steps=pre, val_imgs=vb)        # pretrain (teacher forced)
        before, pack_b = eval_ar(m, vb)
        plot_rollout(f"{name}_before", pack_b)
        dmt(m, kind, steps=dmt_steps)                   # DMT finetune
        after, pack_a = eval_ar(m, vb)
        plot_rollout(f"{name}_after", pack_a)
        res[name] = {"before": before, "after": after}
        print(f"[{name:14s}] nmse_ar {before['nmse_ar']:.3f} -> {after['nmse_ar']:.3f} | "
              f"px_ar {before['px_ar']:.4f} -> {after['px_ar']:.4f} | "
              f"px_enc {before['px_enc']:.4f}")

    names = list(res)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    # per-step AR nMSE before/after
    for name in names:
        ax[0].plot(range(1, T + 1), res[name]["before"]["mse_ar_t"], "--", alpha=0.5)
        ax[0].plot(range(1, T + 1), res[name]["after"]["mse_ar_t"], "-", label=name)
    ax[0].set_xlabel("rollout step"); ax[0].set_ylabel("AR latent nMSE")
    ax[0].set_title("Per-step AR nMSE (dashed=before DMT, solid=after)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # nmse_ar bars before/after
    x = range(len(names)); w = 0.35
    ax[1].bar([i - w / 2 for i in x], [res[n]["before"]["nmse_ar"] for n in names], w, label="before DMT")
    ax[1].bar([i + w / 2 for i in x], [res[n]["after"]["nmse_ar"] for n in names], w, label="after DMT")
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels(names, rotation=15, fontsize=8)
    ax[1].set_ylabel("AR latent nMSE"); ax[1].set_title("AR latent nMSE"); ax[1].legend()
    # px_ar bars before/after
    ax[2].bar([i - w / 2 for i in x], [res[n]["before"]["px_ar"] for n in names], w, label="before DMT")
    ax[2].bar([i + w / 2 for i in x], [res[n]["after"]["px_ar"] for n in names], w, label="after DMT")
    ax[2].set_xticks(list(x)); ax[2].set_xticklabels(names, rotation=15, fontsize=8)
    ax[2].set_ylabel("AR pixel MSE"); ax[2].set_title("AR pixel recon"); ax[2].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/dmt_compare.png", dpi=100); plt.close(fig)
    with open(f"{OUT}/dmt_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nsaved dmt_compare.png + dmt_results.json to", OUT)


if __name__ == "__main__":
    pre = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    ds = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    main(pre, ds)
