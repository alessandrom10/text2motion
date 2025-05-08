import os
import torch
from dataset.dataset import TextToMotionDataset
import h5py
import json

class DFaustDataset(TextToMotionDataset):

    def __init__(self, root_dir = "../data/dfaust"):
        """
        Args:
            root_dir (string): Root directory containing the DFaust dataset
        """
        super().__init__(root_dir)

    def _prepare_samples(self):
        current_id = None
        current_gender = None
        current_sequence = []

        # get current path
        current_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_path, self.root_dir, "subjects_and_sequences.txt")

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line and line[0].isdigit():
                    if current_id is not None:
                        for movement in current_sequence:
                            self.samples.append(((current_id, current_gender, movement), movement))
                    
                    parts = line.split(' (')
                    current_id = int(parts[0])
                    current_gender = parts[1].replace(')', '').strip()
                    current_sequence = []
                
                elif line and current_id is not None:
                    current_sequence.append(line)
            
            if current_id is not None:
                for movement in current_sequence:
                    self.samples.append(((current_id, current_gender, movement), movement))


    def _load_animation(self, animation):
        currentid, currentgender, currentseq = animation
        
        sidseq = str(currentid) + '_' + currentseq

        current_path = os.path.dirname(os.path.abspath(__file__))

        if (currentgender == "male"):
            file_path = os.path.join(current_path, self.root_dir, "registrations_m.hdf5")
        else:
            file_path = os.path.join(current_path, self.root_dir, "registrations_f.hdf5")

        with h5py.File(file_path, 'r') as f:
            verts = f[sidseq][()].transpose([2, 0, 1])
            faces = f['faces'][()]

        # convert verts and faces to torch tensors
        verts = torch.tensor(verts, dtype=torch.float32)
        faces = faces.astype(int)

        # normalize the mesh
        centroid = torch.mean(verts[0], dim=0)
        verts = verts - centroid
        m = torch.max(torch.sqrt(torch.sum(verts**2, dim=1)))
        verts = verts / m

        return verts, faces

    def _load_text(self, text):
        return text
    
    def _load_action_descriptions(self):
        """
        Loads action descriptions from 'action_descriptions.json'
        located in the dataset's root directory.
        """
        current_path = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(current_path, self.root_dir, "action_descriptions.json")
        
        descriptions = {}
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                descriptions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Action descriptions file not found at {json_file_path}. Please ensure the file exists.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {json_file_path}: {e}")
        except Exception as e: # Catch any other unexpected error during loading
            raise RuntimeError(f"Failed to load action descriptions: {e}")

        return descriptions