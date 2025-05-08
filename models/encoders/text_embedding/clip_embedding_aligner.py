import torch
import clip
from typing import Dict, List, Optional
import logging
from .embedding_aligner_base import EmbeddingAligner 

logger = logging.getLogger(__name__)

class CLIPEmbeddingAligner(EmbeddingAligner):
    """
    Concrete implementation using OpenAI's CLIP model via the official 'clip' package.
    Encodes text descriptions into CLIP's text embedding space.
    Aligns input embeddings (e.g., from PointNet) ONLY IF they have been projected
    into the corresponding CLIP embedding space and dimension.
    """
    def __init__(self, 
                    descriptions: Dict[str, str], 
                    model_name: str = "ViT-B/32", 
                    device: Optional[torch.device] = None
                ) -> None:
        """
        Initializes the CLIP aligner, loading the specified CLIP model.

        :param descriptions: Dictionary mapping action names to descriptions.
        :param model_name: Name of the CLIP model (e.g., "ViT-B/32", "RN50"). See clip.available_models().
        :param device: The torch device to use. Auto-detects if None.
        """
        # Determine device early for model loading
        temp_device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing CLIP Aligner with model '{model_name}' on device '{temp_device}'.")

        try:
            self.clip_model, _ = clip.load(model_name, device=temp_device) 
            self.clip_model.eval() # Set to evaluation mode
        except Exception as e:
            logger.error(f"Failed to load CLIP model '{model_name}': {e}")
            raise RuntimeError(f"Failed to load CLIP model '{model_name}'") from e

        super().__init__(descriptions=descriptions, device=temp_device)

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encodes a list of text strings using the loaded CLIP model's text encoder.

        :param texts: A list of text strings to encode.
        :return: A tensor of normalized text embeddings [num_texts, embedding_dim].
        """
        if not texts:
            # Determine output dimension dynamically if possible, otherwise error
            try:
                output_dim = self.clip_model.text_projection.shape[-1]
            except AttributeError:
                try:
                    output_dim = self.clip_model.visual.output_dim
                except AttributeError:
                    raise ValueError("Cannot determine CLIP output dimension for empty text list.")
            return torch.empty((0, output_dim), device=self.device) 
            
        try:
            # Tokenize text batch - use truncate=True for safety
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device) 
            # Encode text using CLIP model - result is already normalized by default
            text_embeddings = self.clip_model.encode_text(text_tokens) 
            return text_embeddings # Shape [num_texts, embed_dim]
        except Exception as e:
            logger.error(f"Error encoding texts with CLIP: {e}")
            raise RuntimeError("CLIP text encoding failed") from e