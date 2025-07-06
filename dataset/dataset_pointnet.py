# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
import trimesh
from bvh import Bvh
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def load_datasets(args):
    train_dataset = TruebonesDataset(npoints=args.npoint, split='train')
    test_dataset = TruebonesDataset(npoints=args.npoint, split='test')
    test_dataset.set_seg_classes(train_dataset.seg_classes)
    test_dataset.set_part_classes(train_dataset.part_classes)
    test_dataset.set_dict_groups(train_dataset.dict_groups)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    return train_dataset, test_dataset, train_loader, test_loader

def find_files_with_substring(folder_path, substring):
    for filename in os.listdir(folder_path):
        if substring in filename and ".npz" in filename:
            return filename
    return False

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class TruebonesDataset(Dataset):

    def __init__(self,root = './data/Truebone_Z-OO', npoints=2500, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        
        classes = {'Crocodile': 39, 'Pteranodon': 44, 'Skunk': 36, 'Lynx': 39, 'Trex': 61, 'Deer': 60, 'Dragon': 143, 'Coyote': 41, 'Tukan': 18, 'Stego': 41, 'Giantbee': 52, 'Dog': 61, 'HermitCrab': 67, 'Tyranno': 69, 'Tricera': 32, 'Raptor2': 63, 'Hound': 45, 'Lion': 32, 'Anaconda': 31, 'Mammoth': 42, 'SpiderG': 62, 'Parrot': 71, 'Jaguar': 46, 'Horse': 79, 'Goat': 33, 'Bear': 76, 'Isopetra': 63, 'SabreToothTiger': 74, 'Eagle': 52, 'Raptor': 36, 'Scorpion': 64, 'PolarBear': 41, 'Scorpion-2': 59, 'Centipede': 84, 'BrownBear': 39, 'Ant': 41, 'Crab': 54, 'FireAnt': 45, 'KingCobra': 19, 'Bat': 48, 'Alligator': 29, 'Hamster': 42, 'Raindeer': 36, 'Comodoa': 65, 'Pirrana': 22, 'Crow': 29, 'Spider': 71, 'Buzzard': 62, 'Pigeon': 9, 'Hippopotamus': 41, 'Turtle': 50, 'Cricket': 54, 'Dog-2': 131, 'Leapord': 48, 'Rhino': 44, 'Rat': 18, 'Roach': 46, 'Gazelle': 42}

        classes_less_50 = [k for k, v in classes.items() if v < 50]

        # === Load CSV ===
        df = pd.read_csv(root + '/TrueboneZ-OO.csv')
        df = df.dropna()
        df.columns = df.columns.str.strip()
        df = df[df['Group'] != 'Group']  # remove empty groups

        # remove the groups that have more than 50 joints
        df = df[df['Group'].isin(classes_less_50)]

        # shuffle the DataFrame
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        self.seg_classes = {}
        self.part_classes = {}

        # take only the first 1000 rows for training
        n_rows = 350
        if split == 'train':
            df = df.iloc[:n_rows]
        elif split == 'test':
            df = df.iloc[n_rows:n_rows+150]

        self.df = df
        self.check_files()

    # Check if all files in the CSV exist, otherwise remove them from the DataFrame
    def check_files(self):
        valid_rows = []
        for index, row in self.df.iterrows():
            folder = row["Group"].strip()
            animation = row["File"].strip().replace("__", "").replace(".BVH", "")
            matched_file = find_files_with_substring(os.path.join(self.root, folder), animation)
            if matched_file:
                valid_rows.append(row)

                # populate seg_classes with the number of joints
                if row["Group"].strip() not in self.seg_classes:
                    data_archive = np.load(os.path.join(self.root, folder, matched_file))
                    joints_key = f'frame_0_joints'
                    self.seg_classes[row["Group"].strip()] = len(data_archive[joints_key]) if joints_key in data_archive else 0
        
        self.dict_groups = list(self.seg_classes.keys())
        self.dict_groups = {group: i for i, group in enumerate(self.dict_groups)}

        # populate part_classes with the indices of the joints
        curr_sum = 0
        for group, num_joints in self.seg_classes.items():
            self.part_classes[self.dict_groups[group]] = np.arange(num_joints) + curr_sum
            curr_sum += num_joints

        self.df = pd.DataFrame(valid_rows)
        return len(valid_rows) == len(self.df)


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
        
        folder = row["Group"].strip()
        animation = row["File"].strip().replace("__", "").replace(".BVH", "")
        with open(os.path.join(self.root, folder,row["File"].strip())) as f:
            animation_bvh = Bvh(f.read())

        matched_file = find_files_with_substring(os.path.join(self.root, folder), animation)
        if not matched_file:
            raise FileNotFoundError(f"File not found for {folder} - {animation}")

        try:
            data_archive = np.load(os.path.join(self.root, folder, matched_file))
        except FileNotFoundError:
            print(f"File {matched_file} not found. Please check the path.")
            return

        frames = data_archive.keys()
        frame = np.random.choice([int(f.split('_')[1]) for f in frames if f.startswith('frame_')])

        vertices_key = f'frame_{frame}_vertices'
        joints_key = f'frame_{frame}_joints'

        if vertices_key not in data_archive or joints_key not in data_archive:
            raise FileNotFoundError(f"Data for frame {frame} not found in file {matched_file}. Available keys: {list(data_archive.keys())}")

        vertices = data_archive[vertices_key]
        joints = data_archive[joints_key]

        cls = self.dict_groups[row['Group'].strip()]


        replace = vertices.shape[0] < self.npoints
        indices = np.random.choice(vertices.shape[0], size=self.npoints, replace=replace)
        vertices = vertices[indices]
        point_set = np.concatenate((vertices, joints), axis=0)
        point_set = pc_normalize(point_set)
        vertices = point_set[:self.npoints]
        joints = point_set[self.npoints:]

        final_joints = torch.zeros(self.get_num_part(), 3)

        arange = self.part_classes[self.dict_groups[row['Group'].strip()]]
        mask = torch.zeros(self.get_num_part(), dtype=bool)
        mask[arange] = True
        final_joints[arange] = torch.tensor(joints, dtype=torch.float32)

        return vertices, cls, mask, final_joints

    def __len__(self):
        return len(self.df)


