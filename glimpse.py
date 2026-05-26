"""Batched glimpse transform for JEPA-style training.

Generates zoomed/translated views of an image batch by accumulating per-step
actions on top of a stored source. All ops are vectorized over the batch via
``F.affine_grid`` + ``F.grid_sample``.

ActionGenerator design note (Q1 — trajectory length):
    Use a fixed ``T = T_max`` for the whole batch. Each sample independently
    picks its own ``t_stop`` ∈ [1, T_max]; deltas after ``t_stop`` are zeros
    (glimpse stays still for the remaining steps). Tensors stay rectangular
    (B, T, ...) and the "random number of steps" intent is preserved
    per-sample.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F



class GlimpseAction:
    """
    An action that changes the glimpse by a delta (log_scale, x, y).
    Takes in log_scale, x, and y from an ML model action output.

    Attributes:
        log_scale:
            0.0 means no zoom.
            log(2) means zoom in 2x.
            log(1/2) means zoom out 2x. -log(2) = log(1/2) since log(1/x) = -log(x)
        x: Horizontal offset in normalized source coords.
        y: Vertical offset in normalized source coords.
    """
    def __init__(self, log_scale: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        assert log_scale.shape == x.shape == y.shape, f"log_scale/x/y shape mismatch: {log_scale.shape}, {x.shape}, {y.shape}"
        
        # The class attributes
        self.log_scale = log_scale
        self.x = x
        self.y = y
        

class Glimpse:
    """
    Absolute glimpse state RELATIVE to the original image.
    
    Glimpse history is aligned by t as follows:
    #   glimpse_states[t] -> state BEFORE action t (log_scale, x, y)
    #   glimpse_actions[t] -> action applied at step t (log_scale, dx, dy)
    glimpse_staetes does not include current state (most recent state after applying an action at time t)
    
    log_scale:
        0.0 means original image scale.
        2.0 means zoomed in e²≈7.39x
        -2.0 means zoomed out e²≈7.39x
    """
    def __init__(
        self,
        image_batch: torch.Tensor, # Original images of shape (B, H, W) or (B, H, W, C). PS: Latter isn't currently supported 
        log_scale: torch.Tensor = torch.Tensor([0]), # Has to be of shape (B, 1) or (1). Initial scale of images.
        x: torch.Tensor = torch.Tensor([0]), # Has to be of shape (B, 1) or (1). Initial x-coords of images.
        y: torch.Tensor = torch.Tensor([0]), # Has to be of shape (B, 1) or (1). Initial y-coords of images.
        T_max: int = 16, # Max trajectory length stored in history buffers.
    ):

        # 1. Getting Batch size        
        self.B = B = image_batch.shape[0]
        
        # 2. Expanding scale, x and y if not correct shape
        if log_scale.shape[0] != B:
            log_scale = log_scale.expand((B, 1))
        if x.shape[0] != B:
            x = x.expand((B, 1))
        if y.shape[0] != B:
            y = y.expand((B, 1))
        
        # 3. Preallocated trajectory buffers, aligned by step t:
        self.glimpse_states = log_scale.new_zeros(T_max, B, 3)
        self.glimpse_actions = log_scale.new_zeros(T_max, B, 3)
        
        # 4. Defining vars
        self.log_scale = log_scale
        self.x = x
        self.y = y
        self.T_max = T_max # Max trajectories
        self.t = 0 # initial time
        self.image_batch = image_batch
        self.original_image_batch = image_batch
        
        
    def initialize_images(self):
        """
        Initializes self.image_batch into random starting positions.
        """
        pass
        

    def apply(self, delta: GlimpseAction):
        """Apply a delta action to current Glimpse state to get a new Glimpse state
        and append current-state and the delta action to history.

        History is a sliding FIFO of length T_max:
            glimpse_states[0]  -> oldest stored state
            glimpse_states[-1] -> most recent state (just appended)
        Once full, each new step shifts the buffer left by one and overwrites
        the last slot. ``self.t`` saturates at T_max (count of valid entries).
        
        Returns new state (in case we want to visualize it or use it).
        """
        current_glimpse_state = torch.cat([self.log_scale, self.x, self.y], dim=-1) # (B, 3)
        new_glimpse_action = torch.cat([delta.log_scale, delta.x, delta.y], dim=-1)  # (B, 3)

        if self.t < self.T_max:
            # Buffer not yet full: write at next free slot
            self.glimpse_states[self.t]  = current_glimpse_state
            self.glimpse_actions[self.t] = new_glimpse_action
            self.t += 1
            
        else:
            # Buffer full: drop oldest, append newest at the end
            self.glimpse_states  = torch.roll(self.glimpse_states,  shifts=-1, dims=0)
            self.glimpse_actions = torch.roll(self.glimpse_actions, shifts=-1, dims=0)
            self.glimpse_states[-1]  = current_glimpse_state
            self.glimpse_actions[-1] = new_glimpse_action

        # Update current absolute state
        self.log_scale = self.log_scale + delta.log_scale
        self.x = self.x + delta.x
        self.y = self.y + delta.y
        
        new_glimpse_state = torch.cat([self.log_scale, self.x, self.y], dim=-1) # (B, 3)
        return new_glimpse_state


    def transform(self, t: int = -1) -> torch.Tensor:
        """
        Render the image batch at glimpse state `t`.
        
        Args:
            t: -1 for the current live state; otherwise an index into
            `glimpse_states` in [0, self.t).
        
        Returns:
            transformed_images: Transformed batch with the same layout 
            as `self.image_batch`. Regions outside the source are zero-padded.
        """
        
        # Getting most current state if t is -1
        if t == -1:
            log_scale = self.log_scale
            x = self.x
            y = self.y
            
        else:
            assert 0 <= t < self.t, f"t={t} out of range [0, {self.t})"
            state = self.glimpse_states[t]  # (B, 3): log_scale, x, y
            log_scale = state[:, 0:1]
            x = state[:, 1:2]
            y = state[:, 2:3]
            
        scale = torch.exp(log_scale)
        
        # Get original image batch of this glimpse and handle different channel sizes
        img = self.original_image_batch
        orig_ndim = img.ndim
        if orig_ndim == 3:  # (B, H, W) mono
            img = img.unsqueeze(1)
        elif orig_ndim == 4:  # (B, H, W, C)
            img = img.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"expected (B,H,W) or (B,H,W,C), got {img.shape}")
        
        B, C, H, W = img.shape
        inv_s = (1.0 / scale).squeeze(-1)  # (B,) - transformation matrix is 1/s
        zero = inv_s.new_zeros(B) # rotation is 0 for now
        
        # Create the transformation matrix (scale and translation)
        theta = torch.stack([
            torch.stack([inv_s, zero,  x.squeeze(-1)], dim=-1),
            torch.stack([zero,  inv_s, y.squeeze(-1)], dim=-1),
        ], dim=1)  # (B, 2, 3)
        
        grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=False)
        transformed_images = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        if orig_ndim == 3:
            return transformed_images.squeeze(1)
        return transformed_images.permute(0, 2, 3, 1).contiguous()
    
    
def rollout(imgs, T_max=16, scale_sensitivity=0.2, translation_sensitivity=0.1, device="cuda"):
    """Roll out a random glimpse trajectory.

    Returns:
        seed:    (B, 1, 28, 28) initial centered frame to seed the autoregressive chain.
        actions: (T_max, B, 3) per-step deltas.
        input_frames: (T_max, B, 1, 28, 28) true frames at steps 0..T_max-1.
        targets: (T_max, B, 1, 28, 28) true frames at steps 1..T_max.
    """
    B = imgs.shape[0]
    glimpse = Glimpse(
        imgs,
        log_scale=torch.zeros(B, 1, device=device),
        x=torch.zeros(B, 1, device=device),
        y=torch.zeros(B, 1, device=device),
        T_max=T_max,
    )

    # Get random glimpse actions (Maybe this should be a method in glimpse action class?)
    d_log_scale = torch.randn(T_max, B, 1, device=device) * scale_sensitivity
    d_x = torch.randn(T_max, B, 1, device=device) * translation_sensitivity
    d_y = torch.randn(T_max, B, 1, device=device) * translation_sensitivity

    # Apply actions to each frame
    frames = [glimpse.transform(t=-1)] # Initial centered frame
    for t in range(T_max):
        glimpse.apply(GlimpseAction(d_log_scale[t], d_x[t], d_y[t]))
        frames.append(glimpse.transform(t=-1))
        
    frames = torch.stack(frames, dim=1) # (B, T_max+1, 28, 28)
    actions = torch.cat([d_log_scale, d_x, d_y], dim=-1).transpose(0, 1) # (B, T_max, 3)

    seed         = frames[:, 0:1].unsqueeze(2)      # (B, 1, 1, 28, 28)
    targets      = frames[:, 1:].unsqueeze(2)       # (B, T_max, 1, 28, 28)
    input_frames = frames[:, :T_max].unsqueeze(2)   # (B, T_max, 1, 28, 28)
    
    return seed, actions, input_frames, targets
    
#================================================
#                TESTING CODE
#================================================
if __name__ == "__main__":
    import torchvision
    from vis_utils import plot_glimpse_frames
    import torchvision.transforms as T

    # torch.manual_seed(8)
    scale_sensitivity = 0.2
    translation_sensitivity = 0.1


    # 1. Get MNIST batch of shape (B, 28, 28)
    ds = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=T.ToTensor())
    imgs, _ = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)))
    imgs = imgs.squeeze(1) # (B, H, W)
    B = imgs.shape[0]

    # 2. Initial glimpse: scale=1, centered
    T_max = 16
    glimpse = Glimpse(imgs, log_scale=torch.zeros(B, 1), x=torch.zeros(B, 1), y=torch.zeros(B, 1), T_max=T_max)

    # 3. Random per-step deltas (small, so the trajectory is visible)
    #    scale must be > 0; sample multiplicatively around 1
    delta_scale = (torch.randn(T_max, B, 1)) * scale_sensitivity
    delta_x = torch.randn(T_max, B, 1) * translation_sensitivity
    delta_y = torch.randn(T_max, B, 1) * translation_sensitivity

    # 4. Apply each delta for each state to get the transformed sequentiall transformer glimpses
    # Render the resulting current state
    frames = []
    for t in range(T_max):
        glimpse.apply(GlimpseAction(delta_scale[t], delta_x[t], delta_y[t]))
        frames.append(glimpse.transform(t=-1).detach().cpu())  # (B, H, W)

    # 5. Visualize: rows = batch item, cols = time step
    plot_glimpse_frames(frames)

