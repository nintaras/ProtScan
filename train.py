import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from src.embeddings import ESM2Embedder
from src.model import ProteinClassifier
from src.dataset import SyntheticProteinDataset, pad_residue_collate

def train_pipeline():
    # 1. Καθορισμός Συσκευής (Device)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"-> Εκπαίδευση σε συσκευή: {device}")

    os.makedirs("assets", exist_ok=True)

    # 2. Αρχικοποίηση ESM-2 Embedder & Dataset
    embedder = ESM2Embedder(device=str(device))
    full_dataset = SyntheticProteinDataset(num_samples=240, seed=42)

    # Split: 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    # DataLoaders με custom collate function για ESM-2 feature padding
    collate_fn = lambda b: pad_residue_collate(b, embedder=embedder, max_len=100)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 3. Μοντέλο, Loss & Optimizer
    model = ProteinClassifier(embed_dim=320, num_classes=3, dropout_rate=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    epochs = 12
    best_val_acc = 0.0

    print("\n--- Έναρξη Εκπαίδευσης ProtScan Attentive Classifier ---")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x_batch, mask_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_batch, mask_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x_batch, mask_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                mask_batch = mask_batch.to(device)
                y_batch = y_batch.to(device)

                logits, _ = model(x_batch, mask_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * x_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.1f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "best_protscan_model.pth")

    print(f"\n-> Η εκπαίδευση ολοκληρώθηκε! Καλύτερο Val Acc: {best_val_acc:.1f}%")
    print("-> Το μοντέλο αποθηκεύτηκε ως 'best_protscan_model.pth'.")

    # 4. Αξιολόγηση & Confusion Matrix
    model.load_state_dict(torch.load("best_protscan_model.pth"))
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_batch, mask_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            logits, _ = model(x_batch, mask_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    class_names = ["Beta-Lactamase (AMR)", "Protein Kinase", "Transmembrane"]
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Σχεδίαση και Αποθήκευση Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='ProtScan: Validation Confusion Matrix',
           ylabel='True Functional Family',
           xlabel='Predicted Functional Family')

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig("assets/confusion_matrix.png", dpi=150)
    print("-> Ο πίνακας σύγχυσης αποθηκεύτηκε στο 'assets/confusion_matrix.png'.")
    plt.show()

if __name__ == "__main__":
    train_pipeline()
