import torch
from typing import Optional, List
import logging
import clip
from .text_encoder_base import TextEncoder 

logger = logging.getLogger(__name__)

class CLIPTextEncoder(TextEncoder):
    """
    Encodes text using an OpenAI CLIP model. Returns normalized embeddings.
    """
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[torch.device] = None) -> None:
        """
        Initializes the CLIP text encoder.

        :param model_name: Name of the CLIP model (e.g., "ViT-B/32", "RN50"). See clip.available_models().
        :param device: The torch device to use. Auto-detects if None.
        """
        super().__init__(device=device) # Sets self.device
        logger.info(f"Initializing CLIPTextEncoder with model '{model_name}' on device '{self.device}'.")
        
        try:
            # Load the CLIP model (only need the model part)
            self.clip_model, _ = clip.load(model_name, device=self.device)
            self.clip_model.eval() # Set to evaluation mode
            
            # Determine and store embedding dimension
            self._embed_dim = self._get_clip_embedding_dim()
            if not self._embed_dim:
                 raise RuntimeError(f"Could not determine embedding dimension for CLIP model '{model_name}'.")
            logger.info(f"CLIP model '{model_name}' loaded. Embedding dimension: {self._embed_dim}.")

        except Exception as e:
            logger.error(f"Failed to load CLIP model '{model_name}': {e}")
            raise RuntimeError(f"Failed to load CLIP model '{model_name}'") from e

    def _get_clip_embedding_dim(self) -> Optional[int]:
        """ Helper to get CLIP output dimension """
        if hasattr(self.clip_model, 'text_projection') and self.clip_model.text_projection is not None:
            # Output dim of the text projection layer
            return self.clip_model.text_projection.shape[-1]
        elif hasattr(self.clip_model, 'visual') and hasattr(self.clip_model.visual, 'output_dim'):
            # Use visual output dim as fallback (should match text)
            return self.clip_model.visual.output_dim
        else:
            logger.warning("Could not reliably determine CLIP embedding dim from model attributes.")
            return None # Indicate failure

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes a single text string using CLIP.

        :param text: The text string.
        :return: Normalized embedding tensor, shape [1, embedding_dim].
        """
        if not text:
            logger.warning("Encoding empty string with CLIP, returning zero vector.")
            return torch.zeros((1, self.embedding_dimension), device=self.device)
            
        try:
            # Tokenize expects a list, even for a single string
            text_token = clip.tokenize([text], truncate=True).to(self.device)
            # Encode text - CLIP returns normalized features
            text_embedding = self.clip_model.encode_text(text_token) 
            return text_embedding # Shape [1, embedding_dim]
        except Exception as e:
            logger.error(f"Error encoding text with CLIP: '{text[:50]}...'. Error: {e}")
            raise RuntimeError("CLIP text encoding failed") from e
            
    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encodes a batch of text strings using CLIP efficiently.

        :param texts: A list of text strings to encode.
        :return: A tensor containing normalized text embeddings, shape [batch_size, embedding_dim].
        """
        if not texts:
            return torch.empty((0, self.embedding_dimension), device=self.device)
        try:
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            text_embeddings = self.clip_model.encode_text(text_tokens)
            return text_embeddings
        except Exception as e:
            logger.error(f"Error batch encoding texts with CLIP: {e}")
            raise RuntimeError("CLIP batch text encoding failed") from e
        
    @torch.no_grad()
    def combine(self, text: str, other_embedding: torch.Tensor) -> torch.Tensor:
        """
        Encodes the text using CLIP and concatenates its embedding with the other_embedding.

        :param text: The text string to encode.
        :param other_embedding: The other embedding tensor (e.g., PointNet) [1, other_dim].
        :return: The concatenated embedding tensor [1, other_dim + clip_embed_dim].
        """
        logger.debug(f"Combining CLIP text embedding for '{text[:30]}...' with other embedding of shape {other_embedding.shape}")
        # 1. Encode the text to get [1, clip_embed_dim]
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