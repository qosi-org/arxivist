# Very Deep Convolutional Networks (arxiv_1409_1556) — VGGNet

Reproduction of VGGNet (Simonyan & Zisserman, ICLR 2015).

## Contents
- `src/vgg.py` — VGG16/19 architectures
- `train.py` — smoke-test loop
- `configs/config.yaml` — hyperparameters

## Run
```bash
python train.py
```

Note: Trained on ImageNet-1K 224×224 RGB images with standard preprocessing.
