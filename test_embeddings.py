from src.embeddings import ESM2Embedder

def test_pipeline():
    embedder = ESM2Embedder()

    # Ενδεικτική πρωτεϊνική αλληλουχία (Beta-lactamase τμήμα)
    sample_seq = "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERF"
    print(f"\nΑλληλουχία δοκιμής (μήκος: {len(sample_seq)} αμινοξέα): {sample_seq[:30]}...")

    seq_rep, res_rep = embedder.embed_sequence(sample_seq)

    print(f"\n[Επιτυχία] Εξαγωγή Embeddings:")
    print(f"-> Sequence Embedding Shape: {seq_rep.shape} (Αναμενόμενο: torch.Size([320]))")
    print(f"-> Residue Embeddings Shape: {res_rep.shape} (Αναμενόμενο: torch.Size([{len(sample_seq)}, 320]))")

if __name__ == "__main__":
    test_pipeline()
