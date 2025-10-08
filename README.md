# PA1: Neural Network & CNN Implementation

**DGIST CSE303 Introduction to Deep Learning - Programming Assignment 1**

**GitHub**: https://github.com/yongjoon2001/DGIST-Deep-Learning-PA1

---

## 📋 Overview

MNIST digit classification using 4 implementations:

1. **3-Layer Neural Network (Pure Python)** - `nn_pure_python.py`
2. **3-Layer Neural Network (PyTorch)** - `nn_framework.py`
3. **3-Layer CNN (Pure Python)** - `cnn_pure_python.py`
4. **3-Layer CNN (PyTorch)** - `cnn_framework.py`

---

## 📁 Project Structure

```
PA1/
├── nn_pure_python.py          # NN Pure Python implementation
├── nn_framework.py            # NN PyTorch implementation
├── cnn_pure_python.py         # CNN Pure Python implementation
├── cnn_framework.py           # CNN PyTorch implementation
├── run_all.py                 # Run all models
│
├── dataset/
│   ├── dataloader.py
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
│
├── checkpoints/               # Saved models (auto-generated)
│   ├── nn_pure_python/
│   ├── nn_framework/
│   ├── cnn_pure_python/
│   └── cnn_framework/
│
└── results/                   # Visualization outputs (auto-generated)
    ├── nn_pure_python/
    │   ├── loss_graph.png
    │   ├── confusion_matrix.png
    │   └── top3_images.png
    ├── nn_framework/
    ├── cnn_pure_python/
    └── cnn_framework/
```

---

## 🏗️ Model Architectures

### 3-Layer Neural Network
```
Input (784) → Linear (128) → ReLU → Linear (64) → ReLU → Linear (10) → SoftMax
```

### 3-Layer CNN
```
Input (28×28×1)
→ Conv (1→16, 3×3) → ReLU → MaxPool (2×2)
→ Conv (16→32, 3×3) → ReLU → MaxPool (2×2)
→ Flatten (1568) → Linear (10) → SoftMax
```

---

## 🚀 Quick Start

### Installation

```bash
pip install numpy matplotlib seaborn torch
```

### Run All Models

```bash
cd PA1
python run_all.py
```

### Run Individual Model

```bash
# Neural Network (Pure Python)
python nn_pure_python.py

# Neural Network (PyTorch)
python nn_framework.py

# CNN (Pure Python)
python cnn_pure_python.py

# CNN (PyTorch)
python cnn_framework.py
```

**⚠️ Important**: Run from the `PA1` directory to ensure proper dataset loading.

---

## 📊 Output Files

Each model generates:

1. **Loss Graph** (`loss_graph.png`)
   - Training and test loss over epochs

2. **Confusion Matrix** (`confusion_matrix.png`)
   - 10×10 normalized probability matrix

3. **Top 3 Images** (`top3_images.png`)
   - Top 3 highest confidence predictions per class

4. **Training Metrics** (`training_metrics.txt`)
   - Model architecture, hyperparameters, and results

---

## 📈 Expected Performance

| Model | Train Acc | Test Acc | Training Time (CPU) |
|-------|-----------|----------|---------------------|
| NN Pure Python | ~97% | ~95% | ~2 min |
| NN PyTorch | ~98% | ~96% | ~30 sec |
| CNN Pure Python | ~95% | ~92% | ~10 min |
| CNN PyTorch | ~99% | ~98% | ~1 min |

---

## 🔧 Implementation Details

### Pure Python Implementation
- **Libraries**: NumPy only (no PyTorch/TensorFlow)
- **Features**:
  - Manual forward/backward propagation
  - Custom gradient computation
  - Manual SGD weight updates

### PyTorch Implementation
- **Libraries**: PyTorch framework
- **Features**:
  - `nn.Module` for model architecture
  - Automatic differentiation (Autograd)
  - GPU acceleration support (CUDA/MPS)
  - Optimized optimizers (Adam, SGD with momentum)

### Device Support
- **CUDA**: NVIDIA GPU (fastest)
- **MPS**: Apple Silicon GPU (M1/M2/M3)
- **CPU**: Universal fallback

---

## 🐛 Troubleshooting

### FileNotFoundError: dataset not found
**Solution**: Run from `PA1` directory:
```bash
cd PA1
python cnn_framework.py
```

### MPS Device Type Mismatch (Mac M1/M2/M3)
```
RuntimeError: Input type (torch.FloatTensor) and weight type (MPSFloatType) should be the same
```
**Solution**: Fixed in v1.4 - tensors are now created directly on the target device.

### Memory Error (CNN Pure Python)
**Solution**: Reduce batch size in the code (16 → 8).

---

## 📝 Assignment Submission

**Due**: October 16, 2024, 14:30 PM
**Submit to**: lms.dgist.ac.kr

**File format**: `PA1_studentID_name.zip`

**Contents**:
1. Source code (4 Python files)
2. PDF report with results:
   - Loss graphs (train & test)
   - 10×10 Confusion matrices
   - Top 3 images per class
   - Comparison of all 4 models

**Note**: Exclude dataset files from submission.

---

## 📚 References

- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/
- **Course**: CSE303 Introduction to Deep Learning
- **Institution**: DGIST

---

## 👨‍💻 Author

**Course**: CSE303 Introduction to Deep Learning
**Institution**: DGIST
**GitHub**: https://github.com/yongjoon2001/DGIST-Deep-Learning-PA1

---

## 📌 Version History

- **v1.4** (2025-10-08): Apple Silicon GPU (MPS) support
- **v1.3** (2025-10-07): Training metrics text file
- **v1.2** (2025-10-07): Improved confusion matrix visualization
- **v1.1** (2025-10-07): Optimizer optimization
