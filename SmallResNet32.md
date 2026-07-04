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

## SmallResNet32 v2

The optimized model keeps the CIFAR-style stem but expands the final residual
stage to the standard ResNet-18 channel progression.

Model changes:
- Width: 64
- Residual channels: 64 -> 128 -> 256 -> 512
- Parameters: 11,169,858
- Dropout: 0.1
- Random erasing: probability 0.15, size 2..6 pixels
- EMA: decay 0.99, enabled after epoch 4
- Prediction: original + horizontal-flip logit averaging

Training performance:
- Train/val split: 90000 / 10000 (same seed and split as v1)
- Epochs: 20
- Batch size: 512
- Best validation accuracy: 0.9769 (epoch 20)
- Best validation loss: 0.165965
- Training time: about 30 minutes, excluding environment setup
- Checkpoint: `outputs_v2/best_model.pt`
- Submission: `submission_v2.csv`

Run:

```powershell
.\.venv\Scripts\python.exe train_aidet.py --device cuda --epochs 20 --batch-size 512 --workers 4 --skip-predict
.\.venv\Scripts\python.exe train_aidet.py --predict-only --checkpoint outputs_v2\best_model.pt --device cuda --batch-size 512 --workers 4
```

## EfficientNet ensemble

To push beyond the 32x32 ResNet ceiling, two ImageNet-pretrained
EfficientNet-B0 models resize the image to 128x128 inside the model. One uses
RGB and the other converts RGB to YCbCr. Both use the same seed-42 train/val
split as v1/v2, so the reported comparison has no cross-split leakage.

Validation results on the common 10,000-image split:
- SmallResNet v2: 97.69%
- EfficientNet-B0 YCbCr: 98.61%
- EfficientNet-B0 RGB: 98.67%
- Calibrated three-model flip-TTA ensemble: 99.09%

The ensemble combines binary logit margins as:

```text
score = ycbcr_margin + 1.94 * rgb_margin + 0.64 * resnet_margin
label = 1 if score > 0.194 else 0
```

Generate the final submission:

```powershell
.\.venv\Scripts\python.exe predict_ensemble.py --device cuda --batch-size 384 --workers 4
```

Artifacts:
- `outputs_effnet_ycbcr/best_model.pt`
- `outputs_effnet_rgb_seed42/best_model.pt`
- `outputs_v2/best_model.pt`
- `submission_ensemble.csv`
