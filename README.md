# PA1: Neural Network & CNN Implementation

**DGIST CSE303 딥러닝개론 Programming Assignment 1**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yongjoon2001/DGIST-Deep-Learning-PA1)

---

## 📋 과제 개요

MNIST 손글씨 숫자 데이터셋을 사용하여 **4가지 딥러닝 모델**을 구현하고 비교 분석합니다:

1. **3-Layer Neural Network (Pure Python)** - `nn_pure_python.py`
2. **3-Layer Neural Network (PyTorch)** - `nn_framework.py`
3. **3-Layer CNN (Pure Python)** - `cnn_pure_python.py`
4. **3-Layer CNN (PyTorch)** - `cnn_framework.py`

### 🎯 학습 목표

- **이론과 실습의 연결**: 강의자료의 수식을 직접 코드로 구현
- **프레임워크 비교**: Pure Python vs PyTorch 성능 및 구현 차이 이해
- **아키텍처 이해**: Fully Connected Network와 CNN의 차이 체득

---

## 🏗️ 모델 아키텍처

### 1️⃣ 3-Layer Neural Network (Fully Connected)

```
Input (784)
  ↓
Linear Layer 1: 784 → 128
  ↓
ReLU Activation
  ↓
Linear Layer 2: 128 → 64
  ↓
ReLU Activation
  ↓
Linear Layer 3: 64 → 10
  ↓
SoftMax (Output: 10 classes)
```

**설계 근거:**
- **784 input**: MNIST 28×28 픽셀을 flatten
- **128 → 64 감소**: 점진적 특징 추출 및 overfitting 방지
- **ReLU**: Gradient vanishing 문제 해결, 빠른 학습
- **10 output**: 숫자 0-9 분류

**학습 설정:**
- Pure Python: Epochs 20, LR 0.1, SGD (manual)
- PyTorch: Epochs 20, LR 0.01, SGD (momentum=0.9)

---

### 2️⃣ 3-Layer CNN (Convolutional Neural Network)

```
Input (28×28×1)
  ↓
Conv Layer 1: 1 → 16 channels, 3×3 kernel, padding=1
  ↓
ReLU + MaxPool (2×2, stride=2) [28×28 → 14×14]
  ↓
Conv Layer 2: 16 → 32 channels, 3×3 kernel, padding=1
  ↓
ReLU + MaxPool (2×2, stride=2) [14×14 → 7×7]
  ↓
Flatten: 32×7×7 = 1568
  ↓
Linear Layer: 1568 → 10
  ↓
SoftMax (Output: 10 classes)
```

**설계 근거:**
- **채널 증가 (1→16→32)**: 계층적 특징 추출 (edge → shape → pattern)
- **3×3 kernel + padding**: VGGNet 스타일, receptive field 유지
- **MaxPooling**: 공간 차원 축소, translation invariance
- **Xavier 초기화**: CNN의 깊이를 고려한 weight 초기화

**학습 설정:**
- Pure Python: Epochs 5, LR 0.01, Batch 16, SGD (manual)
- PyTorch: Epochs 10, LR 0.001, Batch 32, Adam

---

## 📁 프로젝트 구조

```
PA1/
├── 📄 PA1_NN_CNN.pdf              # 과제 설명서
├── 📄 README.md                   # 프로젝트 문서 (본 파일)
│
├── 🐍 Python 구현 파일
│   ├── nn_pure_python.py          # NN Pure Python 구현
│   ├── nn_framework.py            # NN PyTorch 구현
│   ├── cnn_pure_python.py         # CNN Pure Python 구현
│   ├── cnn_framework.py           # CNN PyTorch 구현
│   └── run_all.py                 # 전체 실행 스크립트
│
├── 📂 dataset/                    # MNIST 데이터셋
│   ├── dataloader.py              # 제공된 데이터로더
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
│
├── 📂 checkpoints/                # 학습된 모델 체크포인트 (자동 생성)
│   ├── nn_pure_python/
│   │   ├── model.pkl              # Pure Python 모델 파라미터
│   │   └── training_metrics.txt   # 학습 지표 텍스트 파일
│   ├── cnn_pure_python/
│   │   ├── model.pkl
│   │   └── training_metrics.txt
│   ├── nn_framework/
│   │   ├── model.pth              # PyTorch 모델 체크포인트
│   │   └── training_metrics.txt
│   └── cnn_framework/
│       ├── model.pth
│       └── training_metrics.txt
│
└── 📂 results/                    # 시각화 결과물 (자동 생성)
    ├── nn_pure_python/
    │   ├── loss_graph.png         # Loss 변화 그래프
    │   ├── confusion_matrix.png   # Confusion Matrix
    │   └── top3_images.png        # 클래스별 Top 3 이미지
    ├── nn_framework/
    ├── cnn_pure_python/
    └── cnn_framework/
```

---

## 🚀 실행 방법

### 📦 필요한 라이브러리

```bash
pip install numpy matplotlib seaborn torch
```

**라이브러리 용도:**
- `numpy`: 수치 연산 및 Pure Python 구현
- `matplotlib`: 그래프 시각화
- `seaborn`: Confusion matrix 히트맵
- `torch`: PyTorch 구현 (nn_framework, cnn_framework)

---

### ▶️ 실행 옵션

#### 1. 전체 실행 (권장)

```bash
cd PA1
python run_all.py
```

**실행 순서:**
1. 3-Layer Neural Network (Pure Python)
2. 3-Layer Neural Network (PyTorch)
3. 3-Layer CNN (Pure Python)
4. 3-Layer CNN (PyTorch)

모든 결과가 `results/` 및 `checkpoints/` 폴더에 저장됩니다.

---

#### 2. 개별 실행

```bash
cd PA1

# Neural Network (Pure Python)
python nn_pure_python.py

# Neural Network (PyTorch)
python nn_framework.py

# CNN (Pure Python)
python cnn_pure_python.py

# CNN (PyTorch)
python cnn_framework.py
```

---

## 📊 결과물 상세 설명

### 1️⃣ **Loss Graph** (`loss_graph.png`)

- **내용**: Training Loss와 Test Loss의 에포크별 변화
- **목적**:
  - Overfitting 감지 (Train ↓, Test ↑)
  - 학습 수렴 확인
  - 모델 성능 트렌드 분석

**예시:**
```
Epoch 1: Train Loss 2.3, Test Loss 2.1
Epoch 20: Train Loss 0.1, Test Loss 0.15
```

---

### 2️⃣ **Confusion Matrix** (`confusion_matrix.png`)

- **크기**: 10×10 (클래스 0-9)
- **값**: 정규화된 확률 (0.00 ~ 1.00)
- **대각선**: 정확히 분류된 확률 (높을수록 좋음)
- **비대각선**: 오분류 패턴 (낮을수록 좋음)

**개선 사항:**
- ✅ 모든 셀에 숫자 표시 (수동 annotation)
- ✅ 텍스트 색상 자동 조정 (가독성)
- ✅ 고해상도 (DPI 300)

**해석 예시:**
```
실제 8 → 예측 8: 0.95 (정확)
실제 8 → 예측 3: 0.03 (오분류)
```

---

### 3️⃣ **Top 3 Images** (`top3_images.png`)

- **레이아웃**: 10×3 그리드 (10개 클래스, 각 3개 이미지)
- **선정 기준**: 모델이 가장 높은 confidence로 예측한 이미지
- **표시 정보**:
  - 클래스 번호
  - 순위 (1, 2, 3)
  - Confidence 값

**활용:**
- 모델이 "확신하는" 이미지 패턴 분석
- 클래스별 학습 품질 확인

---

### 4️⃣ **Training Metrics** (`training_metrics.txt`)

**새로 추가된 기능!** 📝

학습 결과를 텍스트 형식으로 저장하여 쉽게 비교 가능:

```txt
============================================================
3-Layer Neural Network (PyTorch) - Training Metrics
============================================================

Model Architecture:
  Input Layer: 784 neurons
  Hidden Layer 1: 128 neurons (ReLU)
  Hidden Layer 2: 64 neurons (ReLU)
  Output Layer: 10 neurons (CrossEntropyLoss)

Training Configuration:
  Epochs: 20
  Learning Rate: 0.01
  Batch Size: 32
  Optimizer: SGD (momentum=0.9)
  Loss Function: CrossEntropyLoss
  Device: CPU

Final Results:
  Train Accuracy: 0.9876 (98.76%)
  Test Accuracy: 0.9654 (96.54%)
  Final Train Loss: 0.0423
  Final Test Loss: 0.1234

Loss History (per epoch):
------------------------------------------------------------
Epoch      Train Loss      Test Loss
------------------------------------------------------------
1          2.3012          2.1543
2          1.8765          1.7234
...
20         0.0423          0.1234
```

**포함 정보:**
- 모델 아키텍처 상세
- 학습 설정 (하이퍼파라미터)
- 최종 정확도 및 손실
- 전체 에포크 Loss 히스토리

---

## 🔬 구현 세부사항

### Pure Python 구현

**특징:**
- **NumPy만 사용**: 프레임워크 없이 순수 구현
- **Forward Propagation**: 수동 행렬 연산
- **Backward Propagation**: Gradient 직접 계산
  ```python
  dW = np.dot(input.T, dout)
  db = np.sum(dout, axis=0, keepdims=True)
  ```
- **Weight Update**: 수동 SGD
  ```python
  W -= learning_rate * dW
  b -= learning_rate * db
  ```

**CNN 추가 구현:**
- Convolution 연산 (4중 for loop)
- MaxPooling (argmax 추적)
- Padding 처리

**장점:**
- 내부 동작 완전 이해
- 디버깅 용이
- 교육적 가치

**단점:**
- 느린 속도 (특히 CNN)
- GPU 미지원

---

### PyTorch 구현

**특징:**
- **torch.nn 모듈**: `nn.Linear`, `nn.Conv2d`, `nn.MaxPool2d`
- **자동 미분**: Autograd 시스템
- **GPU 가속**: CUDA 지원
- **최적화된 Optimizer**: Adam, SGD with Momentum

**코드 예시:**
```python
class ThreeLayerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
```

**장점:**
- 빠른 학습 속도
- 산업계 표준
- 확장성 우수

---

## 📚 강의자료 활용

### Slide03 (DL-Model)
- ✅ Linear layer의 forward/backward 구현
- ✅ Weight matrix 연산: `y = Wx + b`

### Slide05 (DL-Loss)
- ✅ Cross Entropy Loss 구현
- ✅ SoftMax와 Loss의 결합

### Slide06 (DL-Optimization)
- ✅ SGD, Momentum, Adam 적용
- ✅ Learning rate scheduling 개념

### Slide07 (Activation)
- ✅ ReLU 구현 및 gradient 계산
- ✅ Vanishing gradient 문제 해결

### Slide08 (ConvolutionalNeuralNet)
- ✅ Convolution, Pooling 구현
- ✅ Local connectivity, Parameter sharing
- ✅ LeNet-5 변형 아키텍처

---

## 🎯 주요 특징 및 개선사항

### ✨ 핵심 기능

1. **모듈화된 설계**
   - 각 레이어를 독립적인 클래스로 구현
   - 재사용 가능한 컴포넌트

2. **실시간 학습 모니터링**
   - 에포크별 Train/Test Loss 및 Accuracy 출력
   - 배치별 진행 상황 (CNN Pure Python)

3. **체크포인트 시스템**
   - 학습된 모델 자동 저장
   - 재학습 없이 결과 재현 가능

4. **종합적 시각화**
   - Loss graph, Confusion matrix, Top 3 images
   - 고해상도 (DPI 300) PNG 저장

5. **텍스트 기반 리포트**
   - `training_metrics.txt` 자동 생성
   - 하이퍼파라미터부터 결과까지 모든 정보 포함

---

### 🔧 최근 업데이트

#### v1.3 (2025-10-07)
- ✅ Training metrics 텍스트 파일 저장 기능 추가
- ✅ Confusion matrix 모든 셀 숫자 표시 보장
- ✅ 학습 중 정확도 실시간 추적

#### v1.2 (2025-10-07)
- ✅ Confusion matrix 시각화 개선 (수동 annotation)
- ✅ DPI 300으로 고화질 이미지 생성
- ✅ 체크포인트 디렉토리 구조화

#### v1.1 (2025-10-07)
- ✅ nn_framework optimizer 변경 (Adam → SGD + Momentum)
- ✅ Learning rate 최적화
- ✅ 데이터 검증 로직 추가

---

## ✅ 과제 요구사항 충족

| 요구사항 | 구현 여부 | 파일 |
|---------|----------|------|
| 3-Layer NN (Pure Python) | ✅ | `nn_pure_python.py` |
| 3-Layer NN (PyTorch) | ✅ | `nn_framework.py` |
| 3-Layer CNN (Pure Python) | ✅ | `cnn_pure_python.py` |
| 3-Layer CNN (PyTorch) | ✅ | `cnn_framework.py` |
| Training/Test Loss Graph | ✅ | `results/*/loss_graph.png` |
| Confusion Matrix | ✅ | `results/*/confusion_matrix.png` |
| Top 3 Images per Class | ✅ | `results/*/top3_images.png` |
| 모델 체크포인트 | ✅ | `checkpoints/*/model.*` |
| 학습 지표 리포트 | ✅ | `checkpoints/*/training_metrics.txt` |

---

## 📈 예상 성능

| 모델 | Train Acc | Test Acc | 학습 시간 (CPU) |
|------|-----------|----------|----------------|
| NN Pure Python | ~97% | ~95% | ~2분 |
| NN PyTorch | ~98% | ~96% | ~30초 |
| CNN Pure Python | ~95% | ~92% | ~10분 |
| CNN PyTorch | ~99% | ~98% | ~1분 |

*성능은 하드웨어 및 초기 가중치에 따라 다를 수 있습니다.*

---

## 🐛 문제 해결

### 1. Import Error
```bash
ModuleNotFoundError: No module named 'torch'
```
**해결:** `pip install torch`

### 2. 메모리 부족 (CNN Pure Python)
```bash
MemoryError: Unable to allocate array
```
**해결:** Batch size 감소 (16 → 8)

### 3. Confusion Matrix 숫자 안 보임
**해결됨!** v1.2부터 수동 annotation으로 모든 셀 표시

---

## 📖 참고 자료

- **과제 설명서**: `PA1_NN_CNN.pdf`
- **MNIST Dataset**: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- **강의자료**:
  - Slide03: DL-Model
  - Slide05: DL-Loss
  - Slide06: DL-Optimization
  - Slide07: DL-Activation
  - Slide08: ConvolutionalNeuralNet
- **논문**:
  - LeNet-5 (1998)
  - AlexNet (2012)
  - VGGNet (2014)

---

## 👨‍💻 개발 정보

- **과목**: CSE303 딥러닝개론
- **학교**: DGIST
- **GitHub**: [https://github.com/yongjoon2001/DGIST-Deep-Learning-PA1](https://github.com/yongjoon2001/DGIST-Deep-Learning-PA1)

---

## 📝 라이센스

This project is for educational purposes only.

---

**마지막 업데이트**: 2025-10-07
**버전**: 1.3.0
