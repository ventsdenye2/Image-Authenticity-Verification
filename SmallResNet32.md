SmallResNet32 for 32x32 AI-generated image detection.

Model:
- Stem: 3x3 Conv -> BatchNorm -> ReLU, width=48
- Residual layer1: 2 BasicBlocks, 48 channels
- Residual layer2: 2 BasicBlocks, 96 channels, stride=2
- Residual layer3: 2 BasicBlocks, 192 channels, stride=2
- Residual layer4: 2 BasicBlocks, 192 channels, stride=2
- Head: AdaptiveAvgPool2d -> Dropout(0.2) -> Linear(192, 2)

Training:
- Train/val split: 90000 / 10000
- Epochs: 12
- Batch size: 512
- Optimizer: AdamW
- LR: 3e-3
- Weight decay: 1e-2
- Scheduler: CosineAnnealingLR
- Label smoothing: 0.05
- Device: RTX 4060 Laptop GPU, PyTorch 2.5.1+cu121

Result:
- Best validation accuracy: 0.9711
- Training time: about 16m40s
- submission.csv: 20000 rows, IDs 1..20000, labels only 0/1