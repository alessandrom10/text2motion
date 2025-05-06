import os
import torch
from dataset import TextToMotionDataset
import h5py

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

        file_path = os.path.join(self.root_dir, "subjects_and_sequences.txt")

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
        if (currentgender == "male"):
            file_path = os.path.join(self.root_dir, "registrations_m.hdf5")
        else:
            file_path = os.path.join(self.root_dir, "registrations_f.hdf5")

        with h5py.File(file_path, 'r') as f:
            verts = f[sidseq][()].transpose([2, 0, 1])
            faces = f['faces'][()]

        # convert verts and faces to torch tensors
        verts = torch.tensor(verts, dtype=torch.float32)
        faces = faces.astype(int)

        return verts, faces

    def _load_text(self, text):
        return text