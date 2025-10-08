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
├── 🐍 Python Implementation Files (60KB total)
│   ├── nn_pure_python.py          (14KB)  - NN Pure Python implementation
│   ├── nn_framework.py            (12KB)  - NN PyTorch implementation
│   ├── cnn_pure_python.py         (21KB)  - CNN Pure Python implementation
│   ├── cnn_framework.py           (13KB)  - CNN PyTorch implementation
│   └── run_all.py                 (1.6KB) - Run all models at once
│
├── 💾 Dataset (11MB - EXCLUDE from submission)
│   ├── dataloader.py              (2.1KB)  - Data loader (INCLUDE in submission)
│   ├── train-images-idx3-ubyte.gz (9.5MB)  - Training images (60,000)
│   ├── train-labels-idx1-ubyte.gz (28KB)   - Training labels
│   ├── t10k-images-idx3-ubyte.gz  (1.6MB)  - Test images (10,000)
│   └── t10k-labels-idx1-ubyte.gz  (4.4KB)  - Test labels
│
├── 💼 Checkpoints (1.5MB - EXCLUDE from submission)
│   ├── nn_pure_python/
│   │   ├── model.pkl              (856KB)  - Trained weights (NumPy)
│   │   └── training_metrics.txt   (1.6KB)  - Training summary
│   ├── nn_framework/
│   │   ├── model.pth              (430KB)  - PyTorch checkpoint
│   │   └── training_metrics.txt   (1.6KB)
│   ├── cnn_pure_python/
│   │   ├── model.pkl              (161KB)  - CNN weights
│   │   └── training_metrics.txt   (1.1KB)
│   └── cnn_framework/
│       ├── model.pth              (83KB)   - CNN checkpoint
│       └── training_metrics.txt   (1.3KB)
│
└── 📊 Results (1.5MB - USE in PDF report)
    ├── nn_pure_python/
    │   ├── loss_graph.png         (35KB)   - Training/test loss curves
    │   ├── confusion_matrix.png   (219KB)  - 10×10 classification matrix
    │   └── top3_images.png        (101KB)  - Top 3 confident predictions
    ├── nn_framework/
    │   ├── loss_graph.png         (38KB)
    │   ├── confusion_matrix.png   (224KB)
    │   └── top3_images.png        (100KB)
    ├── cnn_pure_python/
    │   ├── loss_graph.png         (38KB)
    │   ├── confusion_matrix.png   (231KB)
    │   └── top3_images.png        (101KB)
    └── cnn_framework/
        ├── loss_graph.png         (37KB)
        ├── confusion_matrix.png   (217KB)
        └── top3_images.png        (99KB)

📄 Documentation
├── README.md                      (4.2KB)  - This file
├── PA1_NN_CNN.pdf                 (435KB)  - Assignment instructions
└── report.pdf                     (TBD)    - Assignment report with results
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
### Memory Error (CNN Pure Python)
**Solution**: Reduce batch size in the code (16 → 8).


## 📚 References

- **MNIST Dataset**: http://yann.lecun.com/exdb/mnist/
- **Course**: CSE303 Introduction to Deep Learning
- **Institution**: DGIST

---

## 👨‍💻 Author: Chon Yongjoon

**Course**: CSE303 Introduction to Deep Learning
**Institution**: DGIST
**GitHub**: https://github.com/yongjoon2001/DGIST-Deep-Learning-PA1

---