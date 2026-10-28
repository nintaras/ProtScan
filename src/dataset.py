import torch
from torch.utils.data import Dataset
import random

class SyntheticProteinDataset(Dataset):
    """
    Dataset παραγωγής και προσομοίωσης πρωτεϊνικών οικογενειών με χαρακτηριστικά βιολογικά μοτίβα:
    - Κλάση 0: Beta-Lactamases / AMR Enzymes (Μοτίβο S-X-X-K / active-site serines)
    - Κλάση 1: Protein Kinases (Μοτίβο V-A-I-K / catalytic loops)
    - Κλάση 2: Transmembrane Receptors (Μοτίβα πλούσια σε υδρόφοβα αμινοξέα L, I, V, F)
    """
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self, num_samples: int = 240, min_len: int = 50, max_len: int = 90, seed: int = 42):
        random.seed(seed)
        self.samples = []
        self.labels = []
        
        motifs = {
            0: ["STFK", "SAVK", "SIQK"],              # Beta-lactamase catalytic serine motifs
            1: ["VAIK", "HRDLK", "DFG"],              # Kinase signature motifs
            2: ["LLLLVV", "IVVALL", "LFLVVI"]         # Hydrophobic membrane patches
        }

        for _ in range(num_samples):
            cls = random.choice([0, 1, 2])
            seq_len = random.randint(min_len, max_len)
            
            # Δημιουργία τυχαίας πρωτεϊνικής ραχοκοκαλιάς
            seq = [random.choice(self.AMINO_ACIDS) for _ in range(seq_len)]
            
            # Εισαγωγή των χαρακτηριστικών βιολογικών μοτίβων
            chosen_motif = random.choice(motifs[cls])
            insert_pos = random.randint(10, seq_len - len(chosen_motif) - 5)
            seq[insert_pos : insert_pos + len(chosen_motif)] = list(chosen_motif)
            
            self.samples.append("".join(seq))
            self.labels.append(cls)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def pad_residue_collate(batch, embedder, max_len=120):
    """
    Custom collate function που εξάγει τα per-residue ESM-2 embeddings 
    και εφαρμόζει padding με attention mask.
    """
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    all_res_embeddings = []
    masks = []

    for seq in sequences:
        _, res_rep = embedder.embed_sequence(seq)  # (L, 320)
        length = res_rep.shape[0]

        if length < max_len:
            pad_tensor = torch.zeros((max_len - length, res_rep.shape[1]))
            padded = torch.cat([res_rep, pad_tensor], dim=0)
            mask = torch.cat([torch.ones(length, dtype=torch.bool), torch.zeros(max_len - length, dtype=torch.bool)])
        else:
            padded = res_rep[:max_len, :]
            mask = torch.ones(max_len, dtype=torch.bool)

        all_res_embeddings.append(padded)
        masks.append(mask)

    batch_x = torch.stack(all_res_embeddings)  # (Batch, max_len, 320)
    batch_mask = torch.stack(masks)            # (Batch, max_len)

    return batch_x, batch_mask, labels
