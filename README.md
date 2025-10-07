# PA1: Neural Network & CNN Implementation

DGIST CSE303 딥러닝개론 Programming Assignment 1

## 과제 개요

MNIST 데이터셋을 사용하여 다음 4가지 구현을 완료합니다:

1. **3-layer Neural Network (순수 Python)** - `nn_pure_python.py`
2. **3-layer Neural Network (PyTorch)** - `nn_framework.py`
3. **3-layer CNN (순수 Python)** - `cnn_pure_python.py`
4. **3-layer CNN (PyTorch)** - `cnn_framework.py`

## 구조

### Neural Network 구조
- Input (28×28) → Linear-ReLU → Linear-ReLU → Linear-SoftMax → Output (10)
- Hidden layer sizes: 128, 64

### CNN 구조
- Input (28×28) → Conv-ReLU-MaxPool → Conv-ReLU-MaxPool → Linear-SoftMax → Output (10)
- Conv layers: 16, 32 channels
- Kernel size: 3×3, MaxPool: 2×2

## 파일 구조

```
PA1/
├── dataset/
│   ├── dataloader.py           # 제공된 데이터로더
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
├── results/                    # 결과물 저장 폴더
│   ├── nn_pure_python/
│   ├── nn_framework/
│   ├── cnn_pure_python/
│   └── cnn_framework/
├── nn_pure_python.py          # NN 순수 Python 구현
├── nn_framework.py             # NN PyTorch 구현
├── cnn_pure_python.py          # CNN 순수 Python 구현
├── cnn_framework.py            # CNN PyTorch 구현
├── run_all.py                  # 모든 구현 실행 스크립트
└── README.md
```

## 실행 방법

### 개별 실행
```bash
cd PA1
python nn_pure_python.py       # NN 순수 Python
python nn_framework.py         # NN PyTorch
python cnn_pure_python.py      # CNN 순수 Python
python cnn_framework.py        # CNN PyTorch
```

### 전체 실행
```bash
cd PA1
python run_all.py
```

## 필요한 라이브러리

- `numpy` - 수치 연산
- `matplotlib` - 그래프 그리기
- `seaborn` - confusion matrix 시각화
- `torch` - PyTorch 구현용

## 결과물

각 구현별로 다음 결과물이 생성됩니다:

1. **Loss Graph** (`loss_graph.png`)
   - Training/Test loss 변화 그래프

2. **Confusion Matrix** (`confusion_matrix.png`)
   - 10×10 분류 확률 매트릭스

3. **Top 3 Images** (`top3_images.png`)
   - 각 클래스별 상위 3개 confidence 이미지

## 구현 세부사항

### 순수 Python 구현
- Numpy만 사용하여 forward/backward propagation 구현
- Linear layer, ReLU, SoftMax, Cross-entropy loss 직접 구현
- CNN의 경우 Convolution, MaxPooling 직접 구현

### PyTorch 구현
- torch.nn 모듈 사용
- 자동 미분 활용
- 효율적인 GPU 연산 지원

## 주요 특징

- **모듈화된 설계**: 각 layer를 독립적인 클래스로 구현
- **Backpropagation**: 순수 Python 구현에서 수동으로 gradient 계산
- **결과 시각화**: 훈련 과정과 성능을 시각적으로 분석
- **비교 분석**: 순수 Python vs Framework 구현 성능 비교

## 과제 요구사항 충족

✅ 3-layer NN without framework (순수 Python)  
✅ 3-layer NN with framework (PyTorch)  
✅ 3-layer CNN without framework (순수 Python)  
✅ 3-layer CNN with framework (PyTorch)  
✅ Training/Test loss 그래프  
✅ Confusion matrix  
✅ Top 3 scored images per class  
✅ 결과 비교 및 분석 가능