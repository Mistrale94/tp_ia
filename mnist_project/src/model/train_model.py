# src/model/train_model.py
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from data_preparation import get_data_loaders
from model import ConvNet

def train(model, train_loader, device, n_epoch=20, perm=torch.arange(0, 784).long()):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    loss_history = []
    for epoch in range(n_epoch):
        running_loss = 0.0
        for i, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            data = data[:, perm]
            data = data.view(-1, 1, 28, 28)
            optimizer.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                avg_loss = running_loss / 100
                print(f'Epoch={epoch}, Step={i+1}: Loss={avg_loss:.4f}')
                loss_history.append(avg_loss)
                running_loss = 0.0
    return loss_history

def test(model, test_loader, device, perm=torch.arange(0, 784).long()):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28*28)
            data = data[:, perm]
            data = data.view(-1, 1, 28, 28)
            logits = model(data)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    test_loss /= total
    accuracy = correct / total
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
    return test_loss, accuracy

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    train_loader, test_loader = get_data_loaders(batch_size=64)

    model = ConvNet(input_size=28*28, n_kernels=32, output_size=10).to(device)

    # Entraîner le modèle
    start_time = time.time()
    train(model, train_loader, device, n_epoch=20)
    cnn_training_time = time.time() - start_time
    print(f"Temps d'entraînement CNN: {cnn_training_time:.2f} secondes")

    # Évaluer le modèle
    test_loss, test_accuracy = test(model, test_loader, device)
    
    # Sauvegarder le modèle
    model_path = "convnet_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Modèle ConvNet sauvegardé sous {model_path}")
