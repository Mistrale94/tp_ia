import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, input_size, n_kernels, output_size):
        super().__init__()
        
        # Définition de l'architecture du réseau
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=n_kernels, out_channels=n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),

            nn.Linear(in_features=n_kernels * 4 * 4, out_features=50),
            nn.ReLU(),

            nn.Linear(in_features=50, out_features=output_size)
        )

    def forward(self, x):
        return self.net(x)

n_kernels = 6
input_size = 28 * 28
output_size = 10 

model = ConvNet(input_size=input_size, n_kernels=n_kernels, output_size=output_size)

print(model)

from tqdm import tqdm  # Progress bar for better visualization

def train(model, train_loader, device, n_epoch=1, perm=torch.arange(0, 784).long()):
    model.train()  # Mettre le modèle en mode entraînement
    optimizer = torch.optim.AdamW(model.parameters())  # Définir l'optimizer
    
    for epoch in range(n_epoch):  # Boucle sur le nombre d'epochs
        running_loss = 0.0
        for i, (data, target) in enumerate(tqdm(train_loader)):  # Boucle sur le loader avec une barre de progression
            data, target = data.to(device), target.to(device)  # Envoyer les données et cibles à l'appareil

            # Appliquer les permutations de pixels par la matrice circulaire de Toeplitz
            data = data.view(-1, 28*28)  # Aplatir les données
            data = data[:, perm]  # Appliquer la permutation
            data = data.view(-1, 1, 28, 28)  # Restructurer les données en format 1x28x28

            optimizer.zero_grad()  # Initialiser les gradients
            logits = model(data)  # Prédiction par le modèle
            loss = nn.cross_entropy(logits, target)  # Calculer le loss
            loss.backward()  # Calculer la rétropropagation
            optimizer.step()  # Mettre à jour les poids
            
            running_loss += loss.item()
            if i % 100 == 99:  # Afficher le loss toutes les 100 itérations
                print(f'Epoch={epoch}, Step={i+1}: Loss={running_loss / 100:.4f}')
                running_loss = 0.0

# Exemple d'utilisation de la fonction train

# Assurez-vous que votre modèle et vos loaders sont définis
# Exemple d'initialisation du modèle
model = ConvNet(input_size=28*28, n_kernels=6, output_size=10)
model.to(device)  # Envoyer le modèle sur l'appareil (GPU ou CPU)

# Supposons que train_loader est défini précédemment
train(model, train_loader, device, n_epoch=1)
