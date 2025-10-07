import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset.dataloader import Dataloader
import os

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Xavier/Glorot initialization
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        self.W = np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros(out_channels)
        
        self.input = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.input = x
        N, C, H, W = x.shape
        
        # Add padding
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x
        
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        out = np.zeros((N, self.out_channels, H_out, W_out))
        
        # Optimized convolution
        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        conv_region = x_padded[n, :, h_start:h_end, w_start:w_end]
                        out[n, c_out, h, w] = np.sum(conv_region * self.W[c_out]) + self.b[c_out]
        
        return out
    
    def backward(self, dout):
        N, C, H, W = self.input.shape
        _, _, H_out, W_out = dout.shape
        
        # Add padding to input for backward pass
        if self.padding > 0:
            x_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            dx_padded = np.zeros_like(x_padded)
        else:
            x_padded = self.input
            dx_padded = np.zeros_like(x_padded)
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Update gradients
                        self.dW[c_out] += x_padded[n, :, h_start:h_end, w_start:w_end] * dout[n, c_out, h, w]
                        dx_padded[n, :, h_start:h_end, w_start:w_end] += self.W[c_out] * dout[n, c_out, h, w]
                
                self.db[c_out] += np.sum(dout[:, c_out, :, :])
        
        # Remove padding from gradient
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
        
        return dx

class MaxPooling:
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.input = None
        self.mask = None
    
    def forward(self, x):
        self.input = x
        N, C, H, W = x.shape
        
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        
        out = np.zeros((N, C, H_out, W_out))
        self.mask = np.zeros_like(x)
        
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        pool_region = x[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(pool_region)
                        out[n, c, h, w] = max_val
                        
                        # Create mask for backward pass
                        mask = (pool_region == max_val)
                        self.mask[n, c, h_start:h_end, w_start:w_end] = mask / np.sum(mask)
        
        return out
    
    def backward(self, dout):
        N, C, H_out, W_out = dout.shape
        dx = np.zeros_like(self.input)
        
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        dx[n, c, h_start:h_end, w_start:w_end] += dout[n, c, h, w] * self.mask[n, c, h_start:h_end, w_start:w_end]
        
        return dx

class LinearLayer:
    def __init__(self, input_size, output_size):
        # Xavier initialization
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        self.b = np.zeros((1, output_size))
        self.input = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.input = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, dout):
        self.dW = np.dot(self.input.T, dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.W.T)

class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class SoftMax:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        x = x - np.max(x, axis=1, keepdims=True)  # For numerical stability
        exp_x = np.exp(x)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out
    
    def backward(self, dout):
        return dout

def cross_entropy_loss(y_pred, y_true):
    batch_size = y_pred.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-7)) / batch_size
    return loss

def cross_entropy_loss_gradient(y_pred, y_true):
    batch_size = y_pred.shape[0]
    return (y_pred - y_true) / batch_size

class ThreeLayerCNN:
    def __init__(self):
        # First conv layer: 1 -> 16 channels
        self.conv1 = ConvLayer(1, 16, 3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPooling(2, stride=2)
        
        # Second conv layer: 16 -> 32 channels
        self.conv2 = ConvLayer(16, 32, 3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPooling(2, stride=2)
        
        # Linear layer: flattened features -> 10 classes
        # After conv1+pool1: 28x28 -> 14x14
        # After conv2+pool2: 14x14 -> 7x7
        # So final size: 32 * 7 * 7 = 1568
        self.linear = LinearLayer(32 * 7 * 7, 10)
        self.softmax = SoftMax()
        
        self.train_losses = []
        self.test_losses = []
    
    def forward(self, x):
        # First conv block
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.maxpool1.forward(out)
        
        # Second conv block
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.maxpool2.forward(out)
        
        # Flatten and linear layer
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)
        out = self.linear.forward(out)
        out = self.softmax.forward(out)
        
        return out
    
    def backward(self, dout):
        # Backward through softmax and linear
        dout = self.softmax.backward(dout)
        dout = self.linear.backward(dout)
        
        # Reshape for conv layers
        dout = dout.reshape(dout.shape[0], 32, 7, 7)
        
        # Backward through second conv block
        dout = self.maxpool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)
        
        # Backward through first conv block
        dout = self.maxpool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.conv1.backward(dout)
    
    def update_weights(self, learning_rate):
        self.conv1.W -= learning_rate * self.conv1.dW
        self.conv1.b -= learning_rate * self.conv1.db
        self.conv2.W -= learning_rate * self.conv2.dW
        self.conv2.b -= learning_rate * self.conv2.db
        self.linear.W -= learning_rate * self.linear.dW
        self.linear.b -= learning_rate * self.linear.db
    
    def train(self, train_loader, test_loader, epochs=10, learning_rate=0.01):
        print("Training 3-layer CNN (Pure Python - Full Implementation)...")
        
        for epoch in range(epochs):
            train_loss = 0
            train_batches = 0
            
            for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):
                # Forward pass
                y_pred = self.forward(batch_images)
                loss = cross_entropy_loss(y_pred, batch_labels)
                train_loss += loss
                train_batches += 1
                
                # Backward pass
                dout = cross_entropy_loss_gradient(y_pred, batch_labels)
                self.backward(dout)
                self.update_weights(learning_rate)
                
                # Print progress every 100 batches
                if batch_idx % 100 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.4f}")
            
            avg_train_loss = train_loss / train_batches
            self.train_losses.append(avg_train_loss)
            
            # Evaluate on test set
            test_loss = self.evaluate(test_loader)
            self.test_losses.append(test_loss)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    def evaluate(self, test_loader):
        test_loss = 0
        test_batches = 0
        
        for batch_images, batch_labels in test_loader:
            y_pred = self.forward(batch_images)
            loss = cross_entropy_loss(y_pred, batch_labels)
            test_loss += loss
            test_batches += 1
        
        return test_loss / test_batches
    
    def get_accuracy(self, loader):
        correct = 0
        total = 0
        
        for batch_images, batch_labels in loader:
            y_pred = self.forward(batch_images)
            predicted = np.argmax(y_pred, axis=1)
            actual = np.argmax(batch_labels, axis=1)
            correct += np.sum(predicted == actual)
            total += batch_labels.shape[0]
        
        return correct / total

def plot_loss_graph(model, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(model.train_losses, label='Train Loss')
    plt.plot(model.test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss (CNN Pure Python)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(model, test_loader, save_path):
    all_predictions = []
    all_labels = []
    
    for batch_images, batch_labels in test_loader:
        y_pred = model.forward(batch_images)
        predicted = np.argmax(y_pred, axis=1)
        actual = np.argmax(batch_labels, axis=1)
        all_predictions.extend(predicted)
        all_labels.extend(actual)
    
    # Debug output
    print(f"Unique predictions: {np.unique(all_predictions)}")
    print(f"Unique actual labels: {np.unique(all_labels)}")
    
    confusion_matrix = np.zeros((10, 10))
    for true, pred in zip(all_labels, all_predictions):
        confusion_matrix[true][pred] += 1
    
    # Normalize and handle division by zero
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix = np.divide(confusion_matrix, row_sums, 
                                out=np.zeros_like(confusion_matrix), 
                                where=row_sums!=0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='.3f', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (CNN Pure Python)')
    plt.savefig(save_path)
    plt.close()
    
    return confusion_matrix

def get_top3_images(model, test_loader, save_path):
    class_scores = [[] for _ in range(10)]
    class_images = [[] for _ in range(10)]
    
    for batch_images, batch_labels in test_loader:
        y_pred = model.forward(batch_images)
        actual_labels = np.argmax(batch_labels, axis=1)
        
        for i in range(batch_images.shape[0]):
            true_label = actual_labels[i]
            confidence = y_pred[i][true_label]
            class_scores[true_label].append(confidence)
            class_images[true_label].append(batch_images[i].squeeze())
    
    fig, axes = plt.subplots(10, 3, figsize=(12, 30))
    
    for class_idx in range(10):
        scores = np.array(class_scores[class_idx])
        images = np.array(class_images[class_idx])
        
        if len(scores) > 0:
            top3_indices = np.argsort(scores)[-3:][::-1]
            
            for rank, idx in enumerate(top3_indices):
                if idx < len(images):
                    axes[class_idx, rank].imshow(images[idx], cmap='gray')
                    axes[class_idx, rank].set_title(f'Class {class_idx}, Rank {rank+1}\nConf: {scores[idx]:.3f}')
                    axes[class_idx, rank].axis('off')
        else:
            for rank in range(3):
                axes[class_idx, rank].axis('off')
                axes[class_idx, rank].set_title(f'Class {class_idx}, Rank {rank+1}\nNo data')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    dataset_path = "dataset"
    train_loader = Dataloader(dataset_path, is_train=True, batch_size=16, shuffle=True)
    test_loader = Dataloader(dataset_path, is_train=False, batch_size=16, shuffle=False)
    
    model = ThreeLayerCNN()
    
    print("Training 3-layer CNN (Pure Python - Full Implementation)...")
    model.train(train_loader, test_loader, epochs=5, learning_rate=0.01)
    
    train_accuracy = model.get_accuracy(train_loader)
    test_accuracy = model.get_accuracy(test_loader)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    results_dir = "results/cnn_pure_python"
    os.makedirs(results_dir, exist_ok=True)
    
    plot_loss_graph(model, f"{results_dir}/loss_graph.png")
    print("Loss graph saved!")
    
    confusion_matrix = plot_confusion_matrix(model, test_loader, f"{results_dir}/confusion_matrix.png")
    print("Confusion matrix saved!")
    
    get_top3_images(model, test_loader, f"{results_dir}/top3_images.png")
    print("Top 3 images saved!")

if __name__ == "__main__":
    main()