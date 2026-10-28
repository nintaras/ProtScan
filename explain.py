import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.embeddings import ESM2Embedder
from src.model import ProteinClassifier
from src.dataset import SyntheticProteinDataset

def run_protein_explainability():
    device = torch.device("cpu")  # CPU για ασφαλή εξαγωγή attention weights
    
    # 1. Φόρτωση ESM-2 και εκπαιδευμένου ταξινομητή
    print("-> Φόρτωση μοντέλων για residue-level explainability...")
    embedder = ESM2Embedder(device="cpu")
    model = ProteinClassifier(embed_dim=320, num_classes=3)
    model.load_state_dict(torch.load("best_protscan_model.pth", map_location=device))
    model.eval()

    # 2. Επιλογή δείγματος Beta-Lactamase (AMR - Class 0)
    dataset = SyntheticProteinDataset(num_samples=50, seed=7)
    sample_seq = None
    true_label = None

    for seq, lbl in dataset:
        if lbl == 0:  # Beta-lactamase
            sample_seq = seq
            true_label = lbl
            break

    # 3. Εξαγωγή per-residue embeddings και attention weights
    _, res_embeddings = embedder.embed_sequence(sample_seq)  # (L, 320)
    seq_len = res_embeddings.shape[0]

    input_tensor = res_embeddings.unsqueeze(0).to(device)  # (1, L, 320)
    mask = torch.ones((1, seq_len), dtype=torch.bool).to(device)

    with torch.no_grad():
        logits, attn_weights = model(input_tensor, mask)
        pred_class = torch.argmax(logits, dim=1).item()

    class_names = ["Beta-Lactamase (AMR)", "Protein Kinase", "Transmembrane"]
    weights = attn_weights.squeeze().numpy()
    
    # Κανονικοποίηση βαρών στο [0, 1] για οπτικοποίηση
    norm_weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-8)

    # 4. Σχεδίαση Attention Profiles
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [2, 1]})

    # Επίπεδο 1: Ράβδοι προσοχής ανά αμινοξύ
    positions = np.arange(1, seq_len + 1)
    bars = ax1.bar(positions, weights, color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Χρωματισμός των κορυφαίων καταλοίπων
    top_indices = np.argsort(weights)[-4:]
    for idx in top_indices:
        bars[idx].set_color('#d62728')

    ax1.set_title(f"ProtScan: Residue-Level Attention Map | Target: {class_names[true_label]} (Pred: {class_names[pred_class]})", 
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel("Attention Weight")
    ax1.set_xlim(0.5, seq_len + 0.5)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Επίπεδο 2: 2D Heatmap της αλληλουχίας
    heatmap_matrix = norm_weights.reshape(1, -1)
    im = ax2.imshow(heatmap_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax2.set_yticks([])
    ax2.set_xticks(np.arange(seq_len))
    ax2.set_xticklabels(list(sample_seq), fontsize=8, family='monospace')
    ax2.set_xlabel("Amino Acid Residue Position", fontsize=10, fontweight='bold')
    ax2.set_title("Sequence Saliency Map (Active-Site Identification)", fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.4, shrink=0.6)
    cbar.set_label("Normalized Relevance", fontsize=9)

    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/residue_attention_map.png", dpi=150)
    print("-> Το γράφημα explainability αποθηκεύτηκε στο 'assets/residue_attention_map.png'.")
    plt.show()

if __name__ == "__main__":
    run_protein_explainability()
