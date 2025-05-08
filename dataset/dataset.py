from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class TextToMotionDataset(Dataset, ABC):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Root directory containing the dataset
        """
        self.root_dir = root_dir
        self.samples = []  # List of (animation, text) tuples
        
        self._prepare_samples()
        self.action_descriptions = self._load_action_descriptions()

    @abstractmethod
    def _prepare_samples(self):
        """
        Abstract method to prepare sample list.
        Must find animation-text pairs in subclasses.
        """
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        animation, text = self.samples[idx]
        
        animation_data = self._load_animation(animation)
        text_data = self._load_text(text)
            
        return animation_data, text_data

    @abstractmethod
    def _load_animation(self, path):
        """
        Load animation data (sequence of meshes/motion)
        Returns tuple of:
            - 'verts': Tensor of shape (num_frames, num_vertices, 3)
            - 'faces': Tensor of shape (num_faces, 3)
        """
        pass

    @abstractmethod
    def _load_text(self, path):
        """
        Load text data
        Returns the text as a string
        """
        pass

    @abstractmethod
    def _load_action_descriptions(self):
        """
        Abstract method to load action descriptions.
        Subclasses must implement this to return a dictionary mapping
        action names (str) to their descriptions (str).
        An empty dictionary should be returned if loading fails.
        """
        pass