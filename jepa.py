"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
import einops
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots: int = 17, num_proj: int = 512):
        """
        Args:
            knots: 
            num_proj:
        """
        super().__init__()
        self.num_proj = num_proj
        
        # Setup the integration grid (generate knots equally spaced sample points in surface domain [0, 3])
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        
        
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        
        # the Gaussian weighting function w(t)
        window = torch.exp(-t.square() / 2.0)
        
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
        A = torch.randn(D, self.num_proj, device=z.device)
        A = A.div_(A.norm(p=2, dim=0))
        
        # compute the epps-pulley statistic
        x_t = (z @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * z.size(-2)
        return statistic.mean() # average over projections and time
    

class JEPA(nn.Module):
    
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        decoder=None,
        projector=None,
        projector_pred=None,
        knots=17,
        num_proj=512
    ):
        super().__init__()
        
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.projector_pred = projector_pred or nn.Identity()
        self.decoder = decoder 
        self.sigreg = SIGReg(knots, num_proj)
        
    def forward(self, image: torch.Tensor, action: torch.Tensor, ar_steps=0):
        """
        Args:
            image(s): (B, T, C, H, W)
            action(s): (B, T, A) # A = 3 for action glimpse
        """
        z_img, z_action = self.encode(image, action) # (B, T, Z_img) and (B, T, Z_action)
        z_pred = self.predict(z_img, z_action, ar_steps=ar_steps)        
        
        return z_pred, z_img, z_action
        
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
    
    
    def predict(self, z_img, z_action, ar_steps=0):
        """
        Predict next state embedding.
        Args:
            z_img: (B, T, D)
            z_action: (B, T, A_embedding)
        """
        preds = self.predictor(z_img, z_action, ar_steps=ar_steps)
        preds = self.projector_pred(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=z_img.size(0)) # unflatten
        return preds
    
    def decode(self, z_img):
        """z_img: (B, T, D) -> (B, T, 1, H, W)."""
        assert self.decoder is not None, "No decoder attached."
        img = self.decoder(z_img)
        return img

    def recon_loss(self, z_img, images):
        """MSE between decoded latents and ground-truth images."""
        recon = self.decode(z_img)
        return F.mse_loss(recon, images.float())
    
    
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
    
        
        
