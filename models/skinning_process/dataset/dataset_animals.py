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

def hierarchy_bvh(percorso_bvh):
    if not os.path.exists(percorso_bvh):
        print(f"Error: File does not exists '{percorso_bvh}'")
        return []

    hierarchy_pairs = []
    parent_stack = []

    try:
        with open(percorso_bvh, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                keyword = parts[0].upper()

                if keyword in ["ROOT", "JOINT"]:
                    bone_name = parts[1]
                    
                    if parent_stack:
                        parent_name = parent_stack[-1]
                        hierarchy_pairs.append((parent_name, bone_name))
                    
                    parent_stack.append(bone_name)

                elif keyword == "END":
                    parent_stack.append("EndSite_Placeholder")

                elif keyword == "}":
                    if parent_stack:
                        parent_stack.pop()

    except Exception as e:
        print(f"Error: {e}")
        return []

    return hierarchy_pairs

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

class DatasetAnimals(Dataset):

    def __init__(self,root = './data/Truebone_Z-OO', npoints=2500, split='train', class_choice=None, normal_channel=False):
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

        # take only the first 350 rows for training
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

        point_set = np.concatenate((vertices, joints), axis=0)
        point_set = pc_normalize(point_set)
        vertices = point_set[:vertices.shape[0]]
        joints = point_set[vertices.shape[0]:]


        joint_hierarchy = hierarchy_bvh(os.path.join(self.root, folder, row["File"].strip()))
        skin_data = np.load(os.path.join(self.root, folder, 'skin_data.npz'), allow_pickle=True)
        weights = skin_data['weights']                   # gt skinning weights, ordered by skin_data['joint_names']
        faces = skin_data['faces']

        return vertices, joints, weights, joint_hierarchy, faces

    def __len__(self):
        return len(self.df)


