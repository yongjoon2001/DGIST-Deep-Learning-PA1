import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset.dataloader import Dataloader
import os
import pickle

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.1
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
        x = x - np.max(x, axis=1, keepdims=True)
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

class ThreeLayerNN:
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10):
        self.layer1 = LinearLayer(input_size, hidden1_size)
        self.relu1 = ReLU()
        self.layer2 = LinearLayer(hidden1_size, hidden2_size)
        self.relu2 = ReLU()
        self.layer3 = LinearLayer(hidden2_size, output_size)
        self.softmax = SoftMax()
        
        self.train_losses = []
        self.test_losses = []
    
    def forward(self, x):
        out = self.layer1.forward(x)
        out = self.relu1.forward(out)
        out = self.layer2.forward(out)
        out = self.relu2.forward(out)
        out = self.layer3.forward(out)
        out = self.softmax.forward(out)
        return out
    
    def backward(self, dout):
        dout = self.softmax.backward(dout)
        dout = self.layer3.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.layer2.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.layer1.backward(dout)
    
    def update_weights(self, learning_rate):
        self.layer1.W -= learning_rate * self.layer1.dW
        self.layer1.b -= learning_rate * self.layer1.db
        self.layer2.W -= learning_rate * self.layer2.dW
        self.layer2.b -= learning_rate * self.layer2.db
        self.layer3.W -= learning_rate * self.layer3.dW
        self.layer3.b -= learning_rate * self.layer3.db
    
    def train(self, train_loader, test_loader, epochs=50, learning_rate=0.01):
        for epoch in range(epochs):
            train_loss = 0
            train_batches = 0
            
            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.reshape(batch_images.shape[0], -1)
                
                y_pred = self.forward(batch_images)
                loss = cross_entropy_loss(y_pred, batch_labels)
                train_loss += loss
                train_batches += 1
                
                dout = cross_entropy_loss_gradient(y_pred, batch_labels)
                self.backward(dout)
                self.update_weights(learning_rate)
            
            avg_train_loss = train_loss / train_batches
            self.train_losses.append(avg_train_loss)
            
            test_loss = self.evaluate(test_loader)
            self.test_losses.append(test_loss)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    def evaluate(self, test_loader):
        test_loss = 0
        test_batches = 0
        
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.reshape(batch_images.shape[0], -1)
            y_pred = self.forward(batch_images)
            loss = cross_entropy_loss(y_pred, batch_labels)
            test_loss += loss
            test_batches += 1
        
        return test_loss / test_batches
    
    def predict(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.forward(x)
    
    def get_accuracy(self, loader):
        correct = 0
        total = 0

        for batch_images, batch_labels in loader:
            batch_images = batch_images.reshape(batch_images.shape[0], -1)
            y_pred = self.forward(batch_images)
            predicted = np.argmax(y_pred, axis=1)
            actual = np.argmax(batch_labels, axis=1)
            correct += np.sum(predicted == actual)
            total += batch_labels.shape[0]

        return correct / total

    def save_model(self, filepath):
        """Save model parameters to file"""
        model_data = {
            'layer1_W': self.layer1.W,
            'layer1_b': self.layer1.b,
            'layer2_W': self.layer2.W,
            'layer2_b': self.layer2.b,
            'layer3_W': self.layer3.W,
            'layer3_b': self.layer3.b,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model parameters from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.layer1.W = model_data['layer1_W']
        self.layer1.b = model_data['layer1_b']
        self.layer2.W = model_data['layer2_W']
        self.layer2.b = model_data['layer2_b']
        self.layer3.W = model_data['layer3_W']
        self.layer3.b = model_data['layer3_b']
        self.train_losses = model_data['train_losses']
        self.test_losses = model_data['test_losses']
        print(f"Model loaded from {filepath}")

def plot_loss_graph(model, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(model.train_losses, label='Train Loss')
    plt.plot(model.test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(model, test_loader, save_path):
    all_predictions = []
    all_labels = []

    for batch_images, batch_labels in test_loader:
        batch_images = batch_images.reshape(batch_images.shape[0], -1)
        y_pred = model.forward(batch_images)
        predicted = np.argmax(y_pred, axis=1)
        actual = np.argmax(batch_labels, axis=1)
        all_predictions.extend(predicted)
        all_labels.extend(actual)

    confusion_matrix = np.zeros((10, 10))
    for true, pred in zip(all_labels, all_predictions):
        confusion_matrix[true][pred] += 1

    # Normalize
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix = np.divide(
        confusion_matrix,
        row_sums,
        out=np.zeros_like(confusion_matrix, dtype=float),
        where=row_sums!=0
    )

    # Visualization
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        square=True,
        linewidths=1,
        linecolor='white',
        cbar_kws={'label': 'Probability', 'shrink': 0.8},
        annot_kws={'size': 10, 'weight': 'normal'},
        vmin=0,
        vmax=1,
        xticklabels=range(10),
        yticklabels=range(10)
    )

    plt.xlabel('Predicted', fontsize=13, weight='bold')
    plt.ylabel('Actual', fontsize=13, weight='bold')
    plt.title('Confusion Matrix', fontsize=15, weight='bold', pad=15)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    return confusion_matrix

def get_top3_images(model, test_loader, save_path):
    class_scores = [[] for _ in range(10)]
    class_images = [[] for _ in range(10)]
    
    for batch_images, batch_labels in test_loader:
        batch_images_flat = batch_images.reshape(batch_images.shape[0], -1)
        y_pred = model.forward(batch_images_flat)
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
        
        top3_indices = np.argsort(scores)[-3:][::-1]
        
        for rank, idx in enumerate(top3_indices):
            if idx < len(images):
                axes[class_idx, rank].imshow(images[idx], cmap='gray')
                axes[class_idx, rank].set_title(f'Class {class_idx}, Rank {rank+1}\nConf: {scores[idx]:.3f}')
                axes[class_idx, rank].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    dataset_path = "dataset"
    train_loader = Dataloader(dataset_path, is_train=True, batch_size=32, shuffle=True)
    test_loader = Dataloader(dataset_path, is_train=False, batch_size=32, shuffle=False)
    
    model = ThreeLayerNN()
    
    print("Training 3-layer Neural Network (Pure Python)...")
    model.train(train_loader, test_loader, epochs=20, learning_rate=0.1)
    
    train_accuracy = model.get_accuracy(train_loader)
    test_accuracy = model.get_accuracy(test_loader)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    results_dir = "results/nn_pure_python"
    os.makedirs(results_dir, exist_ok=True)
    
    plot_loss_graph(model, f"{results_dir}/loss_graph.png")
    print("Loss graph saved!")
    
    confusion_matrix = plot_confusion_matrix(model, test_loader, f"{results_dir}/confusion_matrix.png")
    print("Confusion matrix saved!")
    
    get_top3_images(model, test_loader, f"{results_dir}/top3_images.png")
    print("Top 3 images saved!")

    # Save model
    model.save_model("checkpoints/nn_pure_python/model.pkl")

if __name__ == "__main__":
    main()