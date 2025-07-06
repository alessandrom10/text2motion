import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import trimesh
from bvh import Bvh
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def load_datasets(args):
    train_dataset = TruebonesDataset(npoints=args.npoint, split='train')
    test_dataset = TruebonesDataset(npoints=args.npoint, split='test')

    test_dataset.set_seg_classes(train_dataset.seg_classes)
    test_dataset.set_part_classes(train_dataset.part_classes)
    test_dataset.set_dict_groups(train_dataset.dict_groups)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10
    )
    return train_dataset, test_dataset, train_loader, test_loader

def find_files_with_substring(folder_path, substring):
    return next((f for f in os.listdir(folder_path) if substring in f and f.endswith(".npz")), False)

def pc_normalize(pc):
    pc -= np.mean(pc, axis=0)
    return pc / np.max(np.linalg.norm(pc, axis=1))

class TruebonesDataset(Dataset):
    def __init__(self, root='./data/Truebone_Z-OO', npoints=2500, split='train', perc_test=0.15):
        self.root = root
        self.npoints = npoints
        self.split = split

        self.classes = {
            k: v for k, v in {
                'Crocodile': 39, 'Pteranodon': 44, 'Skunk': 36, 'Lynx': 39, 'Trex': 61,
                'Deer': 60, 'Dragon': 143, 'Coyote': 41, 'Tukan': 18, 'Stego': 41,
                'Giantbee': 52, 'Dog': 61, 'HermitCrab': 67, 'Tyranno': 69, 'Tricera': 32,
                'Raptor2': 63, 'Hound': 45, 'Lion': 32, 'Anaconda': 31, 'Mammoth': 42,
                'SpiderG': 62, 'Parrot': 71, 'Jaguar': 46, 'Horse': 79, 'Goat': 33,
                'Bear': 76, 'Isopetra': 63, 'SabreToothTiger': 74, 'Eagle': 52,
                'Raptor': 36, 'Scorpion': 64, 'PolarBear': 41, 'Scorpion-2': 59,
                'Centipede': 84, 'BrownBear': 39, 'Ant': 41, 'Crab': 54, 'FireAnt': 45,
                'KingCobra': 19, 'Bat': 48, 'Alligator': 29, 'Hamster': 42, 'Raindeer': 36,
                'Comodoa': 65, 'Pirrana': 22, 'Crow': 29, 'Spider': 71, 'Buzzard': 62,
                'Pigeon': 9, 'Hippopotamus': 41, 'Turtle': 50, 'Cricket': 54, 'Dog-2': 131,
                'Leapord': 48, 'Rhino': 44, 'Rat': 18, 'Roach': 46, 'Gazelle': 42
            }.items()
        }

        self.df = pd.read_csv(os.path.join(root, 'TrueboneZ-OO.csv')).dropna()
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df[self.df['Group'] != 'Group']
        self.df = self.df[self.df['Group'].isin(self.classes)]
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        split_idx = int(len(self.df) * (1 - perc_test))
        if split == 'train':
            self.df = self.df.iloc[:split_idx]
        elif split == 'test':
            self.df = self.df.iloc[split_idx:]

        self.seg_classes = {}
        self.part_classes = {}

        self.check_files()

    def check_files(self):
        valid_rows = []
        for _, row in self.df.iterrows():
            group = row["Group"].strip()
            anim = row["File"].strip().replace("__", "").replace(".BVH", "")
            matched = find_files_with_substring(os.path.join(self.root, group), anim)
            if matched:
                valid_rows.append(row)
                if group not in self.seg_classes:
                    archive = np.load(os.path.join(self.root, group, matched))
                    key = 'frame_0_joints'
                    self.seg_classes[group] = len(archive[key]) if key in archive else 0

        self.dict_groups = {g: i for i, g in enumerate(self.seg_classes.keys())}

        curr = 0
        for group, njoints in self.seg_classes.items():
            self.part_classes[self.dict_groups[group]] = np.arange(njoints) + curr
            curr += njoints

        self.df = pd.DataFrame(valid_rows)

    def set_seg_classes(self, seg_classes):
        self.seg_classes = seg_classes

    def set_part_classes(self, part_classes):
        self.part_classes = part_classes

    def set_dict_groups(self, dict_groups):
        self.dict_groups = dict_groups

    def get_num_classes(self):
        return len(self.seg_classes)

    def get_num_part(self):
        return sum(self.seg_classes.values())

    def __getitem__(self, index):
        row = self.df.iloc[index]
        group = row['Group'].strip()
        anim_id = row['File'].strip().replace("__", "").replace(".BVH", "")
        with open(os.path.join(self.root, group, row['File'].strip())) as f:
            _ = Bvh(f.read())

        npz_file = find_files_with_substring(os.path.join(self.root, group), anim_id)
        archive = np.load(os.path.join(self.root, group, npz_file))

        frame_ids = [int(k.split('_')[1]) for k in archive.keys() if k.startswith('frame_')]
        frame = np.random.choice(frame_ids)

        verts_key = f'frame_{frame}_vertices'
        joints_key = f'frame_{frame}_joints'

        verts = archive[verts_key]
        joints = archive[joints_key]

        replace = verts.shape[0] < self.npoints
        idx = np.random.choice(verts.shape[0], self.npoints, replace=replace)
        verts = verts[idx]

        points = np.concatenate((verts, joints), axis=0)
        points = pc_normalize(points)
        verts, joints = points[:self.npoints], points[self.npoints:]

        group_id = self.dict_groups[group]
        joint_indices = self.part_classes[group_id]

        mask = torch.zeros(self.get_num_part(), dtype=torch.bool)
        mask[joint_indices] = True

        final_joints = torch.zeros(self.get_num_part(), 3)
        final_joints[joint_indices] = torch.tensor(joints, dtype=torch.float32)

        return verts, group_id, mask, final_joints

    def __len__(self):
        return len(self.df)