import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import argparse

def generate_sbert_embeddings(input_csv: str, output_embed_path: str, sbert_model_name: str):
    df = pd.read_csv(input_csv)
    
    if 'text' not in df.columns:
        raise ValueError("The input CSV must contain a 'text' column.")
    
    texts = df['text'].astype(str).tolist()
    model = SentenceTransformer(sbert_model_name)

    print(f"Generating embeddings for {len(texts)} sentences...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
    
    embedding_dict = {text: emb.cpu() for text, emb in zip(texts, embeddings)}
    torch.save(embedding_dict, output_embed_path)
    
    print(f"SBERT embeddings saved to '{output_embed_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SBERT embeddings from a CSV file.")
    parser.add_argument('--input', type=str, required=True, help="Path to the CSV file with a 'text' column.")
    parser.add_argument('--output_embed', type=str, required=True, help="Path to save the .pt file with embeddings.")
    parser.add_argument('--sbert_model', type=str, default='all-MiniLM-L6-v2', help="Name of the SBERT model.")

    args = parser.parse_args()

    generate_sbert_embeddings(
        input_csv=args.input,
        output_embed_path=args.output_embed,
        sbert_model_name=args.sbert_model
    )
