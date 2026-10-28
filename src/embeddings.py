import torch
from transformers import AutoTokenizer, AutoModel

class ESM2Embedder:
    """
    Εξαγωγή πρωτεϊνικών αναπαραστάσεων (embeddings) με χρήση του ESM-2 (Meta AI).
    Υποστηρίζει εξαγωγή τόσο σε επίπεδο αμινοξέος (per-residue) όσο και σε επίπεδο
    ολόκληρης αλληλουχίας (sequence-level mean pooling).
    """
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"-> Φόρτωση ESM-2 ({model_name}) σε συσκευή: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_sequence(self, sequence: str):
        """
        Εξάγει embeddings για μία μεμονωμένη πρωτεϊνική αλληλουχία (π.χ. 'MKTIIALSYIFCLVFA...').
        
        Returns:
            seq_embedding: Tensor μεγέθους (embedding_dim,) - Mean pooled αναπαράσταση
            residue_embeddings: Tensor μεγέθους (seq_len, embedding_dim) - Ανά αμινοξύ
        """
        inputs = self.tokenizer(
            sequence, 
            return_tensors="pt", 
            add_special_tokens=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # outputs.last_hidden_state: (batch_size, seq_len + 2, embedding_dim)
            hidden_states = outputs.last_hidden_state.squeeze(0)  # (seq_len + 2, embedding_dim)

            # Αφαιρούμε τα special tokens: <cls> (αρχή) και <eos> (τέλος)
            residue_embeddings = hidden_states[1:-1, :]  # (seq_len, embedding_dim)

            # Sequence-level embedding μέσω mean-pooling των αμινοξέων
            seq_embedding = torch.mean(residue_embeddings, dim=0)  # (embedding_dim,)

        return seq_embedding.cpu(), residue_embeddings.cpu()

    def embed_batch(self, sequences: list[str]):
        """
        Εξάγει mean-pooled embeddings για batch αλληλουχιών με attention mask.
        
        Returns:
            batch_embeddings: Tensor μεγέθους (batch_size, embedding_dim)
        """
        inputs = self.tokenizer(
            sequences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024,
            add_special_tokens=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # (B, L, D)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (B, L, 1)

            # Εξαιρούμε τα padding tokens από το mean pooling
            masked_hidden = last_hidden * attention_mask
            sum_hidden = torch.sum(masked_hidden, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            batch_embeddings = sum_hidden / sum_mask

        return batch_embeddings.cpu()
