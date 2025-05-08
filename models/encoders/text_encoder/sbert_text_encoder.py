import torch
from typing import Optional, List
import logging
from sentence_transformers import SentenceTransformer
from .text_encoder_base import TextEncoder

logger = logging.getLogger(__name__)

class SentenceBERTEncoder(TextEncoder):
    """
    Encodes text using a SentenceTransformer model.
    Normalizes embeddings for consistent output.
    """
    DEFAULT_MODEL = 'all-mpnet-base-v2' # 768 dim

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[torch.device] = None) -> None:
        """
        Initializes the Sentence-BERT text encoder.

        :param model_name: Name of the SentenceTransformer model.
        :param device: The torch device to use. Auto-detects if None.
        """
        super().__init__(device=device) # Sets self.device ('cpu' or 'cuda')
        logger.info(f"Initializing SentenceBERTEncoder with model '{model_name}' on device '{self.device}'.")

        try:
            # Pass device string ('cpu' or 'cuda') to SentenceTransformer
            self.sbert_model = SentenceTransformer(model_name, device=str(self.device))
            # Get embedding dimension from the loaded model
            self._embed_dim = self.sbert_model.get_sentence_embedding_dimension()
            if not self._embed_dim:
                raise RuntimeError(f"Could not determine embedding dimension for SBERT model '{model_name}'.")
            # Verify actual device used by the model, update self.device if needed
            actual_device_str = str(self.sbert_model.device)
            if actual_device_str != str(self.device):
                logger.warning(f"SBERT model loaded on '{actual_device_str}' instead of requested '{self.device}'. Updating.")
                self.device = self.sbert_model.device
                 
            logger.info(f"SBERT model '{model_name}' loaded. Embedding dimension: {self._embed_dim}.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
            raise RuntimeError(f"Failed to load SentenceTransformer model '{model_name}'") from e

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes a single text string using SentenceTransformer and normalizes the result.

        :param text: The text string.
        :return: Normalized embedding tensor, shape [1, embedding_dim].
        """
        if not text:
            logger.warning("Encoding empty string with SBERT, returning zero vector.")
            return torch.zeros((1, self.embedding_dimension), device=self.device)
            
        try:
            # Encode returns tensor shape [1, embedding_dim] when input is list of 1 string
            embedding = self.sbert_model.encode(
                [text], 
                convert_to_tensor=True, 
                device=self.device, 
                show_progress_bar=False
            )
            # Normalize the embedding for consistency
            embedding = embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-9)
            return embedding 
        except Exception as e:
            logger.error(f"Error encoding text with SentenceTransformer: '{text[:50]}...'. Error: {e}")
            raise RuntimeError("SentenceTransformer text encoding failed") from e

    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encodes a batch of text strings using SentenceTransformer efficiently and normalizes.

        :param texts: A list of text strings to encode.
        :return: A tensor containing normalized text embeddings, shape [batch_size, embedding_dim].
        """
        if not texts:
             return torch.empty((0, self.embedding_dimension), device=self.device)
        try:
            embeddings = self.sbert_model.encode(
                texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False # Often disable for batch processing
            )
            # Normalize embeddings
            embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-9)
            return embeddings.to(self.device)
        except Exception as e:
            logger.error(f"Error batch encoding texts with SentenceTransformer: {e}")
            raise RuntimeError("SentenceTransformer batch text encoding failed") from e
        
    @torch.no_grad()
    def combine(self, text: str, other_embedding: torch.Tensor) -> torch.Tensor:
        """
        Encodes the text using SentenceTransformer and concatenates its embedding 
        with the other_embedding.

        :param text: The text string to encode.
        :param other_embedding: The other embedding tensor (e.g., PointNet) [1, other_dim].
        :return: The concatenated embedding tensor [1, other_dim + sbert_embed_dim].
        """
        logger.debug(f"Combining SBERT text embedding for '{text[:30]}...' with other embedding of shape {other_embedding.shape}")
        # 1. Encode the text to get [1, sbert_embed_dim]
        text_embedding = self.encode(text) 

        # 2. Ensure other_embedding is on the same device
        other_embedding = other_embedding.to(self.device)

        # 3. Validate shapes (expecting batch size 1 for both)
        if not (text_embedding.shape[0] == 1 and other_embedding.shape[0] == 1):
            raise ValueError(f"Both embeddings must have batch size 1 for concatenation, "
                             f"got text shape {text_embedding.shape} and other shape {other_embedding.shape}")

        # 4. Concatenate along the feature dimension (dim=1)
        combined_embedding = torch.cat((other_embedding, text_embedding), dim=1)
        logger.debug(f"Combined embedding shape: {combined_embedding.shape}")
        
        return combined_embedding