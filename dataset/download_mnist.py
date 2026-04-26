from pathlib import Path
from torchvision import datasets

ROOT = Path(__file__).resolve().parent

train = datasets.MNIST(root=ROOT, train=True, download=True)
test = datasets.MNIST(root=ROOT, train=False, download=True)

print(f"train: {len(train)} samples")
print(f"test:  {len(test)} samples")
print(f"saved to {ROOT / 'MNIST'}")
