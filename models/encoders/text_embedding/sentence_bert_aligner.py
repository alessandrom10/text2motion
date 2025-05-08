import torch
from typing import Dict, List, Optional
import logging
from sentence_transformers import SentenceTransformer
from .embedding_aligner_base import EmbeddingAligner 

logger = logging.getLogger(__name__)

class SentenceBERTAligner(EmbeddingAligner):
    """
    Concrete implementation using the sentence-transformers library.
    Encodes text descriptions using a chosen SBERT model.
    Aligns input embeddings ONLY IF they have been projected into the SAME 
    embedding space and dimension as the SBERT model output.
    """
    DEFAULT_MODEL = 'all-mpnet-base-v2' # 768 dim

    def __init__(self, 
                    descriptions: Dict[str, str], 
                    model_name: str = DEFAULT_MODEL, 
                    device: Optional[torch.device] = None
                ) -> None:
        """
        Initializes the Sentence-BERT aligner. Requires 'sentence-transformers'.

        :param descriptions: Dictionary mapping action names to descriptions.
        :param model_name: Name of the SentenceTransformer model.
        :param device: The torch device to use. Auto-detects if None.
        """
        temp_device_str = str(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Initializing SentenceBERT Aligner with model '{model_name}' on device '{temp_device_str}'.")
        
        try:
            self.sbert_model = SentenceTransformer(model_name, device=temp_device_str)
            self._actual_device: torch.device = self.sbert_model.device # Get actual device
            logger.info(f"SentenceTransformer model '{model_name}' loaded on device '{self._actual_device}'.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
            raise RuntimeError(f"Failed to load SentenceTransformer model '{model_name}'") from e

        # Determine embedding dimension for internal use if needed by _encode_texts fallback
        self._sbert_embed_dim: int = self.sbert_model.get_sentence_embedding_dimension()
        if not self._sbert_embed_dim:
             raise RuntimeError(f"Could not determine embedding dimension for model '{model_name}'.")
        logger.info(f"Detected SBERT embedding dimension: {self._sbert_embed_dim}")
        
        super().__init__(descriptions=descriptions, device=self._actual_device)
        # Verify the dimension consistency
        if self.embed_dim != self._sbert_embed_dim:
            logger.warning(f"Dimension mismatch: SBERT model reports {self._sbert_embed_dim}, base class computed {self.embed_dim} from encodings.")

    # Correctly implement the abstract method from the base class
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encodes a list of text strings using the loaded SentenceTransformer model.

        :param texts: A list of text strings to encode.
        :return: A tensor of text embeddings [num_texts, embedding_dim]. 
                 Normalization depends on the specific SBERT model. Base class ensures final normalization.
        """
        if not texts:
            return torch.empty((0, self._sbert_embed_dim), device=self.device)
            
        try:
            # Encode the batch of texts. Result is typically normalized for cosine sim models.
            text_embeddings = self.sbert_model.encode(
                texts, 
                convert_to_tensor=True, 
                device=self.device, 
                show_progress_bar=False 
            ) 
            # Return the embeddings; base class _precompute handles final normalization
            return text_embeddings.to(self.device) # Shape [num_texts, embed_dim]
        except Exception as e:
            logger.error(f"Error encoding texts with SentenceTransformer: {e}")
            raise RuntimeError("SentenceTransformer text encoding failed") from e