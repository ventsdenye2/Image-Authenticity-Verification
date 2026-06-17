# Image-Authenticity-Verification

机器学习课程大作业：判断 32x32 图像是真实图像还是 AI 生成图像。

## 方案

使用一个面向小图像的轻量残差 CNN（SmallResNet）做二分类：

- `train/REAL` -> label `0`
- `train/FAKE` -> label `1`
- 自动按类别分层划分验证集
- 训练时使用随机水平翻转、轻微平移、AdamW、Cosine 学习率
- 如果当前 PyTorch 支持 CUDA，会自动使用 GPU 和混合精度
- 训练结束后生成 `submission.csv`

## 运行

```powershell
python train_aidet.py --device auto --epochs 20 --batch-size 512
```

如果已经确认环境支持 CUDA，可以强制使用 GPU：

```powershell
python train_aidet.py --device cuda --epochs 20 --batch-size 512
```

本机已在 `ml-exp3-cnn` 环境安装 CUDA 版 PyTorch，可直接运行：

```powershell
conda run -n ml-exp3-cnn python -u train_aidet.py --device cuda --epochs 12 --batch-size 512 --workers 0
```

只用已有权重重新生成提交文件：

```powershell
python train_aidet.py --predict-only --checkpoint outputs/best_model.pt
```

最终提交文件为：

```text
submission.csv
```
