import torch
from abc import ABC, abstractmethod
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class TextEncoder(ABC):
    """
    Abstract base class for encoding text into embedding vectors.
    """
    def __init__(self, device: Optional[torch.device] = None) -> None:
        """
        Initializes the text encoder.

        :param device: The torch device ('cuda' or 'cpu'). Auto-detects if None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # Logger info moved to concrete implementations for more context
        self._embed_dim: Optional[int] = None # Concrete class __init__ should set this

    @property
    def embedding_dimension(self) -> int:
        """ 
        Returns the output dimension of the embeddings produced by this encoder.
        :return: The embedding dimension.
        :raises AttributeError: If the dimension was not set by the subclass.
        """
        if self._embed_dim is None:
            raise AttributeError(f"{self.__class__.__name__} embedding dimension not set.")
        return self._embed_dim

    @abstractmethod
    @torch.no_grad() # Enforce no_grad for encoding methods
    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes a single text string into an embedding vector.

        :param text: The text string to encode.
        :return: A tensor containing the text embedding, shape [1, embedding_dim]. 
                 Normalization depends on the implementing class (CLIP/SBERT often normalize).
        """
        pass

    @torch.no_grad() # Enforce no_grad for encoding methods
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encodes a batch of text strings. 
        Provides a default implementation by iterating encode(). 
        Subclasses should override this for efficiency if their underlying model supports batching well.

        :param texts: A list of text strings to encode.
        :return: A tensor containing text embeddings, shape [batch_size, embedding_dim].
        """
        if not texts:
             # Return empty tensor with correct dimension if known
            dim = self.embedding_dimension if self._embed_dim is not None else 0
            return torch.empty((0, dim), device=self.device)
             
        logger.debug(f"Encoding batch of {len(texts)} texts individually...")
        # Default implementation iterates, assumes encode returns [1, D]
        embeddings_list = [self.encode(text) for text in texts] 
        
        if not embeddings_list: # Should only happen if texts was empty or all texts failed encode?
            dim = self.embedding_dimension if self._embed_dim is not None else 0
            return torch.empty((0, dim), device=self.device)
            
        return torch.cat(embeddings_list, dim=0) # Combine [1, D] tensors into [N, D]
    
    @abstractmethod
    def combine(self, text: str, other_embedding: torch.Tensor) -> torch.Tensor:
        """
        Encodes the given text and combines its embedding with another provided embedding.

        :param text: The text string to encode.
        :param other_embedding: The other embedding tensor (e.g., from PointNet), 
                                shape [1, other_dim]. Must be on the same device 
                                or will be moved by the implementation.
        :return: A tensor representing the combined embedding. The shape and meaning
                 depend on the combination strategy implemented by the subclass 
                 (e.g., concatenation).
        """
        pass