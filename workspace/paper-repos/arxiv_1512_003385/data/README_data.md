# Dataset — CIFAR-10

CIFAR-10 is fully public (Krizhevsky, 2009) — 60,000 32x32 color images in 10 classes,
50,000 train + 10,000 test.

torchvision will download CIFAR-10 automatically the first time `CIFAR10DataModule` is
instantiated with `download=true`. The default location is `./data/cifar10/`.

To download manually without running training, use `data/download.py`:

    python data/download.py --dest ./data/cifar10

This downloads the standard `cifar-10-python.tar.gz` (~170 MB) from
`https://www.cs.toronto.edu/~kriz/cifar.html` and extracts it into the destination directory.
