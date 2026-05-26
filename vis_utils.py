import matplotlib.pyplot as plt

def plot_glimpse_frames(glimpse_frames, max_batch: int = 3):
    """
    Unroll all glimpses and plot them in a matrix.
    Each row is an unrolled sequence of glimpses at some state given an action a.
    """
    B, T_max = glimpse_frames.shape[:2]
    
    if B < max_batch:
        max_batch=B
    
    # Visualize: rows = batch item, cols = time step
    fig, axes = plt.subplots(max_batch, T_max, figsize=(T_max * 1.0, max_batch * 1.0))
    for b in range(min(max_batch, B)):
        for t in range(T_max):
            ax = axes[b, t]
            ax.imshow(glimpse_frames[b, t].numpy(), cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
    plt.suptitle("rows = batch, cols = t (after applying delta t)")
    plt.tight_layout()
    plt.show()