from sentence_transformers import SentenceTransformer
import torch
import logging
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pandas as pd
import re
from torch.utils.data import Dataset
from typing import Tuple, List

def get_verb_from_filename(filename: str) -> str:
    return re.sub(r'\d+$|\.BVH$|^__', '', filename, flags=re.IGNORECASE).upper()

def load_embeddings_from_pt_file(path: str, expected_rows: int) -> torch.Tensor:
    data = torch.load(path, map_location='cpu', weights_only=True)

    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, dict):
        tensor = next((v for v in data.values() if isinstance(v, torch.Tensor)), None)
        if tensor is None:
            raise ValueError(f"No tensor found in {path}")
    else:
        raise TypeError(f"Unsupported type {type(data)} in {path}")

    if tensor.dim() == 1:
        return tensor
    if tensor.shape[1] == expected_rows:
        logging.warning(f"Tensor shape {tensor.shape} appears transposed. Transposing.")
        tensor = tensor.T
    return tensor

class BaseSemanticActionDataset(Dataset):
    def __init__(self, csv_path: str, embeddings_path: str):
        self.metadata = pd.read_csv(csv_path)
        self.embeddings = load_embeddings_from_pt_file(embeddings_path, len(self.metadata)).float()
        self.metadata['verb'] = self.metadata['File'].apply(get_verb_from_filename)
        
        logging.info("Pre-calculating semantic verb similarities for caching...")
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        unique_verbs = self.metadata['verb'].unique().tolist()
        verb_embeddings = sbert_model.encode(unique_verbs, convert_to_numpy=True)

        similarity_matrix = cosine_similarity(verb_embeddings)
        self.idx_to_verb = self.metadata['verb'].to_dict()

        self.positive_candidates = {verb: [] for verb in unique_verbs}
        self.negative_candidates = {verb: [] for verb in unique_verbs}
        
        verb_indices_map = self.metadata.groupby('verb').groups

        for i, verb1 in enumerate(unique_verbs):
            for j, verb2 in enumerate(unique_verbs):
                sim = similarity_matrix[i, j]
                if sim > 0.5:
                    self.positive_candidates[verb1].extend(verb_indices_map[verb2])
                elif sim < 0.3:
                    self.negative_candidates[verb1].extend(verb_indices_map[verb2])
        
        logging.info("Semantic similarity cache created.")

    def __len__(self) -> int:
        return len(self.metadata)

    def _get_positive_sample(self, anchor_index: int, anchor_verb: str) -> torch.Tensor:
        candidates = self.positive_candidates[anchor_verb]
        valid_candidates = [idx for idx in candidates if idx != anchor_index]
        
        if valid_candidates:
            return self.embeddings[random.choice(valid_candidates)]
        else:
            return self.embeddings[anchor_index]

    def _get_negative_sample(self, anchor_verb: str) -> torch.Tensor:
        candidates = self.negative_candidates[anchor_verb]
        if candidates:
            return self.embeddings[random.choice(candidates)]
        else:
            possible_negatives = self.metadata[self.metadata['verb'] != anchor_verb]
            return self.embeddings[random.choice(possible_negatives.index)]

class ActionSimilarityDataset(BaseSemanticActionDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_verb = self.idx_to_verb[index]
        anchor_emb = self.embeddings[index]
        pos_emb = self._get_positive_sample(index, anchor_verb)
        neg_emb = self._get_negative_sample(anchor_verb)
        return anchor_emb, pos_emb, neg_emb

class ActionPairDataset(BaseSemanticActionDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor_verb = self.idx_to_verb[index]
        anchor_emb = self.embeddings[index]
        pos_emb = self._get_positive_sample(index, anchor_verb)
        return anchor_emb, pos_emb

def collate_pairs_for_infonce(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
    anchors, positives = zip(*batch)
    return [torch.stack(list(anchors) + list(positives), dim=0)]

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, mask_prob: float = 0.0):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.mask_prob = mask_prob
        if mask_prob > 0:
            logging.info(f"Classifier-Free Guidance enabled with mask probability: {mask_prob}")

    def _maybe_mask(self, batch):
        if self.mask_prob > 0 and random.random() < 0.5:
            return [t.clone().masked_fill(torch.rand_like(t) < self.mask_prob, 0.0) for t in batch]
        return batch

    def run(self, loader, epochs: int, save_path: str):
        best_loss = float('inf')
        logging.info("Starting training...")
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
                batch = [b.to(self.device) for b in batch]
                masked = self._maybe_mask(batch)
                output = [self.model(b) for b in masked]
                loss = self.loss_fn(*output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            logging.info(f"Epoch {epoch}/{epochs} -> Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), save_path)

        logging.info("Training complete.")

class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate_similarity(self, original_anchor, original_positive, original_negative):
        logging.info("--- Running Evaluation ---")
        with torch.no_grad():
            o_anchor = original_anchor.to(self.device).unsqueeze(0)
            o_positive = original_positive.to(self.device).unsqueeze(0)
            o_negative = original_negative.to(self.device).unsqueeze(0)
            e_anchor = self.model(o_anchor).cpu()
            e_positive = self.model(o_positive).cpu()
            e_negative = self.model(o_negative).cpu()

        sim_orig_pos = cosine_similarity(o_anchor.cpu(), o_positive.cpu())[0][0]
        sim_orig_neg = cosine_similarity(o_anchor.cpu(), o_negative.cpu())[0][0]
        logging.info(f"Original Similarity (Anchor-Positive): {sim_orig_pos:.4f}")
        logging.info(f"Original Similarity (Anchor-Negative): {sim_orig_neg:.4f}")

        sim_enh_pos = cosine_similarity(e_anchor, e_positive)[0][0]
        sim_enh_neg = cosine_similarity(e_anchor, e_negative)[0][0]
        logging.info(f"Enhanced Similarity (Anchor-Positive): {sim_enh_pos:.4f}")
        logging.info(f"Enhanced Similarity (Anchor-Negative): {sim_enh_neg:.4f}")
        logging.info("--- Evaluation Finished ---")

class EnhancementService:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def enhance(self, original_embedding: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            original_tensor = torch.from_numpy(original_embedding).float().to(self.device).unsqueeze(0)
            enhanced_tensor = self.model(original_tensor)
        return enhanced_tensor.cpu().squeeze(0).numpy()