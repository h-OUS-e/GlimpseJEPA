"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
import einops
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class SIGReg(torch.nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer
    Good vis about sigreg: https://the-puzzler.github.io/?p=practical-notes-on-lejepa
    """

    def __init__(self, knots: int = 17, num_proj: int = 512):
        """
        Args:
            knots: 
            num_proj:
        """
        super().__init__()
        self.num_proj = num_proj
        
        # 2. Integration Points
        # Setup the integration grid (generate knots equally spaced sample points in surface domain [0, 3])
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)        
        
        # 3. Theoretical Gaussian CF, the Gaussian weighting function w(t)
        window = torch.exp(-0.5 * t**2)
        
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        
   
        
        # Attach tensors to the SigReg class with the right device using register_buffer()
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        
        # premultiply trapezoid weights by w(t) to avoid multiplying every forward inference (saves computation)
        self.register_buffer("weights", weights * window) 

    def forward(self, z):
        """
        For each timestep, SIGReg takes the (B, D) slice and 
        tests whether those B embeddings form a Gaussian in R^D space.
        
        Args:
            proj: (T, B, D)
        """
        # sample random projections
        D = z.size(-1) # size of latent vector
        
        # 1. Projection (The Observer)
        # Project channels down to sketch_dim
        A = torch.randn(D, self.num_proj, device=z.device)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)
        
        # 4. Empirical CF, compute the epps-pulley statistic
        # proj: [N, sketch_dim] 
        x_t = (z @ A).unsqueeze(-1) * self.t # (T, B, num_proj, num_t)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square() # (T, num_proj, knots)
        statistic = (err @ self.weights) * z.size(-2) # (T, num_proj)
        return statistic.mean() # average over projections and time (a scalar)
    

class JEPA(nn.Module):
    
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        memory_encoder=None,
        memory_predictor=None,
        decoder=None,
        projector=None,
        projector_pred=None,
        encode_memory=True,
        knots=17,
        num_proj=512
    ):
        super().__init__()
        
        self.encoder = encoder
        self.memory_encoder = memory_encoder
        self.predictor = predictor
        self.memory_predictor = memory_predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.projector_pred = projector_pred or nn.Identity()
        self.decoder = decoder 
        self.sigreg = SIGReg(knots, num_proj)
        
    def forward(self, image: torch.Tensor, action: torch.Tensor, z_memory: torch.Tensor | None = None, ar_steps=0):
        """
        Args:
            image(s): (B, T, C, H, W)
            action(s): (B, T, A) # A = 3 for action glimpse
            z_memory: (B, T, Z) Previous latent memory stored
        """
        
        z_img, z_action = self.encode(image, action) # (B, T, Z_img) and (B, T, Z_action)

        # build a running memory latent from the encoded frames
        if self.memory_predictor is not None:
            z_memory = self.predict_memory(z_img, z_memory)

        # predict the next state
        z_pred = self.predict(z_img, z_action, z_memory, ar_steps=ar_steps)

        return z_pred, z_img, z_action


    def predict_memory(self, z_img, z_memory=None):
        """Build the running memory latent from encoded frames.

        Args:
            z_img: (B, T, Z_img) encoded frame latents
            z_memory: (B, hidden) optional initial memory seed, defaults to zeros
        Returns:
            z_memory: (B, T, Z_mem) memory at each step (step t saw frames <= t)
        """
        return self.memory_predictor(z_img, z_memory)

        
    def encode(self, img, action=None):
        """
        Encode observations and actions into embeddings.
        We don't need to encode auto-regressively. We can encode all at once.
        """        
        img = img.float() # (B, T, C, H, W)
        B = img.size(0)
        img = rearrange(img, "b t ... -> (b t) ...") # flatten for encoding to shape (B*T, C, H, W)
        
        z_img = self.encoder(img) # (B*T, C*H*W)
        
        # # Use this when using a transformer only
        # z_img = output.last_hidden_state[:, 0] # z is the cls token here of shape (B*T, D_encoder)
        
        # remap embedding from encoder representation space to the predictor representation space
        z_img = self.projector(z_img) # shape (B*T, D_predictor)
        z_img = rearrange(z_img, "(b t) d -> b t d", b=B) # shape (B, T, D_predictor)
        
        # encode action
        if action is not None:
            z_action = self.action_encoder(action) # (B, T, A_embedding)
        else:
            z_action = None
        
        return z_img, z_action
    
    # def predict(self, z_img, z_action, ar_steps=0):
    #     """
    #     Predict next state embedding.
    #     Args:
    #         z_img: (B, T, D)
    #         z_action: (B, T, A_embedding)
    #     """
    #     preds = self.predictor(z_img, z_action)
    #     preds = self.projector_pred(rearrange(preds, "b t d -> (b t) d"))
    #     preds = rearrange(preds, "(b t) d -> b t d", b=z_img.size(0)) # unflatten
    #     return preds
    
    def predict(self, z_img, z_action, z_memory, ar_steps=0):
        """
        Predict next state embeddings.

        ar_steps=0   -> teacher forcing (parallel pass over ground-truth z_img)
        ar_steps=K   -> first K steps autoregressive, remaining teacher-forced
        ar_steps>=T  -> full autoregressive rollout (no teacher forcing; use at eval)

        Each AR step projects the predictor output back into z_img space via
        projector_pred before feeding it back, so the fed-back state matches the
        space of the encoder's embeddings (and of z_target in the loss).

        Args:
            z_img: (B, T, D)
            z_action: (B, T, A_embedding)
        """
        T = z_img.size(1)

        # Condition the predictor on action + memory. NOTE: cat(action, memory)
        # may dilute the action signal -- alternatives to test: (a) separate AdaLN
        # streams for action vs memory, (b) memory as a prepended sequence token,
        # (c) add memory into the predictor input x.
        cond = z_action if z_memory is None else torch.cat([z_action, z_memory], dim=-1)

        # Option A: Teacher forcing: single parallel pass over the true embeddings
        if not ar_steps or ar_steps==0:
            z_preds = self.predictor(z_img, cond)
            z_preds = self.project(z_preds)
            return z_preds

        # Option B: No teacher forcing up until AR_steps
        ar_steps = min(ar_steps, T)
        z_in = z_img[:, :1] # true first frame as the seed
        preds = []
        for t in range(ar_steps):
            raw = self.predictor(z_in, cond[:, :t + 1])[:, -1:] # predict frame t+1
            pred = self.project(raw)
            preds.append(pred)
            z_in = torch.cat([z_in, pred], dim=1) # feed prediction back as next input
        preds = torch.cat(preds, dim=1) # (B, ar_steps, D)

        if ar_steps >= T:
            return preds

        # Tail: teacher-forced on the remaining true frames, conditioned on the AR prefix
        z_full = torch.cat([z_in, z_img[:, ar_steps + 1:]], dim=1) # length T
        tail = self.project(self.predictor(z_full, cond)[:, ar_steps:])
        return torch.cat([preds, tail], dim=1)

    def project(self, preds):
        """
        Map predictor outputs (B, t, d) back into z_img space via projector_pred.
        This is mainly for transformer architecture to avoid layer normalization
        effect on SigReg. Layer normalization doesn't allow us to project embeddings
        into a gaussian-like distribution effectively according to the paper.   
        More about this can be found here: https://the-puzzler.github.io/?p=practical-notes-on-lejepa     
        """
        B = preds.size(0)
        preds = self.projector_pred(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=B) # unflatten
    
    def decode(self, z_img):
        """z_img: (B, T, D) -> (B, T, 1, H, W)."""
        assert self.decoder is not None, "No decoder attached."
        img = self.decoder(z_img)
        return img
    
    def topk_mse(self, pred, target, frac=0.2):
        # pred/target: (B, T, 1, H, W)
        err = (pred - target.float()).square()
        err = err.flatten(start_dim=2)  # (B, T, pixels)
        k = max(1, int(frac * err.size(-1)))
        return err.topk(k, dim=-1).values.mean()

    def recon_loss(self, z_img, images):
        """MSE between decoded latents and ground-truth images."""
        recon = self.decode(z_img)
        mse = F.mse_loss(recon, images.float())
        # topk_mse = self.topk_mse(recon, images.float(), frac=0.2)
        recon_loss = mse #+ 0.5 * topk_mse
        return recon_loss
    
    
    def mse_last_step(self, z_pred, z_target, mean=True):
        """
        Compute the cost between predicted embeddings and target embeddings
        of the last state only.
        
        Args:
            z_pred: (B, S, T, D) or (B, T, D)
            z_target: (B, S, T, D) or (B, T, D)        
        """
        z_target = z_target[..., -1:, :] # gets last target latent vector (the last horizon step)

        # return last-step loss per action candidate
        if mean:
            loss = F.mse_loss(z_pred[..., -1:, :], z_target.detach(), reduction="mean")
        else:
            loss = F.mse_loss(z_pred[..., -1:, :], z_target.detach(), reduction="none")
            loss = einops.reduce(loss, "... t d -> ...", "sum") # sum loss to shape (B, S) or (B,)

        return loss
    
    
    def mse(self, z_pred, z_target, mean=True):
        """
        Compute cost between predicted and target embeddings for
        each state.
        
        Args:
            z_pred: (B, T, D)
            z_target: (B, T, D)
        """       
        # return loss for each action candidate
        if mean:
            loss = F.mse_loss(z_pred, z_target.detach(), reduction="mean")
        else: 
            loss = F.mse_loss(z_pred, z_target.detach(), reduction="none")
            loss = einops.reduce(loss, "b ... -> b", "sum") # (B,) sum loss on all state, per batch
            loss = loss.mean()

        return loss
    
    
    def mse_weighted(self, z_pred, z_target, mean=None):
        """
        Compute cost between predicted and target embeddings for
        each state.
        
        Args:
            z_pred: (B, T, D)
            z_target: (B, T, D)
        """      
        B, T, D = z_pred.size() 
        
        # return loss for each action candidate
        loss = F.mse_loss(z_pred, z_target.detach(), reduction="none")
        loss = einops.reduce(loss, "... t d -> ... t", "sum") # (B, T) sum loss per state
        weights = 1-(torch.arange(T))/T
        weights = weights.expand(B, T).to(loss.device)
        loss = einops.reduce(loss*weights, "... t -> ... ", "sum") # (B, ) sum weighted loss across states
        loss = loss.mean() # average over batch to get scalar for backprop
        
        return loss
    
    
    def sigreg_loss(self, z_enc):
        """
        Apply SigReg loss to latent vector (usually from the encoder)
        
        Args:
            z_enc: (B, T, D)
        """
        # Transpoze z to apply SigReg per timestep
        z_enc = rearrange(z_enc, "b t ... -> t b ...") # (T, B, D)
        loss = self.sigreg(z_enc)
        
        return loss
    
        
        
