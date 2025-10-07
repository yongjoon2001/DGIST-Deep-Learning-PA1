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
        train_correct = 0
        train_total = 0

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

            # Calculate accuracy
            predicted = torch.argmax(outputs, dim=1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        avg_train_loss = train_loss / train_batches
        train_accuracy = train_correct / train_total
        model.train_losses.append(avg_train_loss)

        model.eval()
        test_loss = 0
        test_batches = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_images, batch_labels in test_loader:
                batch_images = torch.FloatTensor(batch_images).to(device)
                batch_labels = torch.FloatTensor(batch_labels).to(device)

                outputs = model(batch_images)
                targets = torch.argmax(batch_labels, dim=1)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                test_batches += 1

                # Calculate accuracy
                predicted = torch.argmax(outputs, dim=1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()

        avg_test_loss = test_loss / test_batches
        test_accuracy = test_correct / test_total
        model.test_losses.append(avg_test_loss)

        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}')

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

    # Normalize
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix_norm = np.divide(
        confusion_matrix,
        row_sums,
        out=np.zeros_like(confusion_matrix, dtype=float),
        where=row_sums!=0
    )

    # Visualization
    plt.figure(figsize=(14, 12))

    ax = sns.heatmap(
        confusion_matrix_norm,
        annot=False,  # Turn off default annotation
        cmap='Blues',
        square=True,
        linewidths=1.5,
        linecolor='white',
        cbar_kws={'label': 'Probability'},
        vmin=0,
        vmax=1,
        xticklabels=range(10),
        yticklabels=range(10)
    )

    # Manually add text annotations to ensure all values are visible
    for i in range(10):
        for j in range(10):
            text_color = "white" if confusion_matrix_norm[i, j] > 0.5 else "black"
            ax.text(j + 0.5, i + 0.5, f'{confusion_matrix_norm[i, j]:.2f}',
                   ha="center", va="center",
                   color=text_color,
                   fontsize=11, weight='bold')

    plt.xlabel('Predicted', fontsize=14, weight='bold')
    plt.ylabel('Actual', fontsize=14, weight='bold')
    plt.title('Confusion Matrix (CNN PyTorch)', fontsize=16, weight='bold', pad=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return confusion_matrix_norm

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

    # Check data
    print("Checking data...")
    for batch_images, batch_labels in train_loader:
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Images range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
        print(f"Labels shape: {batch_labels.shape}")
        print(f"Label sample: {np.argmax(batch_labels[:5], axis=1)}")
        break

    model = ThreeLayerCNN()

    print("\nTraining 3-layer CNN (PyTorch)...")
    train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.001)
    
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

    # Save model
    checkpoint_dir = "checkpoints/cnn_framework"
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': model.train_losses,
        'test_losses': model.test_losses
    }, f"{checkpoint_dir}/model.pth")
    print(f"Model saved to {checkpoint_dir}/model.pth")

if __name__ == "__main__":
    main()