import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset.dataloader import Dataloader
import os

class ThreeLayerCNN(nn.Module):
    def __init__(self):
        super(ThreeLayerCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.linear = nn.Linear(32 * 7 * 7, 10)
        
        self.train_losses = []
        self.test_losses = []
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)
        x = self.linear(x)  # Remove softmax - CrossEntropyLoss includes it
        
        return x

def train_model(model, train_loader, test_loader, epochs=20, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_images, batch_labels in train_loader:
            batch_images = torch.FloatTensor(batch_images).to(device)
            batch_labels = torch.FloatTensor(batch_labels).to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_images)
            
            targets = torch.argmax(batch_labels, dim=1)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        model.train_losses.append(avg_train_loss)
        
        model.eval()
        test_loss = 0
        test_batches = 0
        
        with torch.no_grad():
            for batch_images, batch_labels in test_loader:
                batch_images = torch.FloatTensor(batch_images).to(device)
                batch_labels = torch.FloatTensor(batch_labels).to(device)
                
                outputs = model(batch_images)
                targets = torch.argmax(batch_labels, dim=1)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                test_batches += 1
        
        avg_test_loss = test_loss / test_batches
        model.test_losses.append(avg_test_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

def get_accuracy(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = torch.FloatTensor(batch_images).to(device)
            batch_labels = torch.FloatTensor(batch_labels).to(device)
            
            outputs = model(batch_images)
            predicted = torch.argmax(outputs, dim=1)
            actual = torch.argmax(batch_labels, dim=1)
            
            correct += (predicted == actual).sum().item()
            total += batch_labels.size(0)
    
    return correct / total

def plot_loss_graph(model, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(model.train_losses, label='Train Loss')
    plt.plot(model.test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss (CNN PyTorch)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(model, test_loader, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = torch.FloatTensor(batch_images).to(device)
            batch_labels = torch.FloatTensor(batch_labels).to(device)
            
            outputs = model(batch_images)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            actual = torch.argmax(batch_labels, dim=1).cpu().numpy()
            
            all_predictions.extend(predicted)
            all_labels.extend(actual)
    
    confusion_matrix = np.zeros((10, 10))
    for true, pred in zip(all_labels, all_predictions):
        confusion_matrix[true][pred] += 1
    
    # Debug output
    print(f"Unique predictions: {np.unique(all_predictions)}")
    print(f"Unique actual labels: {np.unique(all_labels)}")
    
    # Normalize and handle division by zero
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix = np.divide(confusion_matrix, row_sums, 
                                out=np.zeros_like(confusion_matrix), 
                                where=row_sums!=0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='.3f', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (CNN PyTorch)')
    plt.savefig(save_path)
    plt.close()
    
    return confusion_matrix

def get_top3_images(model, test_loader, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    class_scores = [[] for _ in range(10)]
    class_images = [[] for _ in range(10)]
    
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images_tensor = torch.FloatTensor(batch_images).to(device)
            batch_labels_tensor = torch.FloatTensor(batch_labels).to(device)
            
            outputs = model(batch_images_tensor)
            outputs_softmax = F.softmax(outputs, dim=1)  # Apply softmax for confidence
            actual_labels = torch.argmax(batch_labels_tensor, dim=1).cpu().numpy()
            
            for i in range(batch_images.shape[0]):
                true_label = actual_labels[i]
                confidence = outputs_softmax[i][true_label].cpu().item()
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
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    dataset_path = "dataset"
    train_loader = Dataloader(dataset_path, is_train=True, batch_size=32, shuffle=True)
    test_loader = Dataloader(dataset_path, is_train=False, batch_size=32, shuffle=False)
    
    model = ThreeLayerCNN()
    
    print("Training 3-layer CNN (PyTorch)...")
    train_model(model, train_loader, test_loader, epochs=5, learning_rate=0.001)
    
    train_accuracy = get_accuracy(model, train_loader)
    test_accuracy = get_accuracy(model, test_loader)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    results_dir = "results/cnn_framework"
    os.makedirs(results_dir, exist_ok=True)
    
    plot_loss_graph(model, f"{results_dir}/loss_graph.png")
    print("Loss graph saved!")
    
    confusion_matrix = plot_confusion_matrix(model, test_loader, f"{results_dir}/confusion_matrix.png")
    print("Confusion matrix saved!")
    
    get_top3_images(model, test_loader, f"{results_dir}/top3_images.png")
    print("Top 3 images saved!")

if __name__ == "__main__":
    main()