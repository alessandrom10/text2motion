# text2motion

## Dataset Setup

To get started, create a folder named `data` in the root directory of the project. Inside this folder, place the datasets you intend to use.

### DFAUST Dataset

To use the DFAUST dataset:

1. Download it from the official website: [https://dfaust.is.tue.mpg.de/index.html](https://dfaust.is.tue.mpg.de/index.html).
2. Inside the `data` folder, include the following files:

   * `subjects_and_sequences.txt` (available in the DFAUST `scripts` folder)
   * `registrations_f.hdf5`
   * `registrations_m.hdf5`

Make sure the file structure is organized as follows:

```
text2motion/
├── data/
│   ├── dfaust/
│       ├── subjects_and_sequences.txt
│       ├── registrations_f.hdf5
│       └── registrations_m.hdf5
```

### HumanML3D Dataset

To use HumanML3D dataset:
1. Download it from the following link: [HumanML3D](https://polimi365-my.sharepoint.com/:u:/g/personal/11016435_polimi_it/ERuJYO7DDyRBl7PwChzVZZsBXui5ir0K3pEEb-WHMH_4yA?e=R5rN3g)

2. Create the folder `HumanML3D` into the `data` folder.

3. Extract the two folders (joints and texts) into the `HumanML3D` folder.

4. Visualize animations using the script in `visualize_skeleton_animation.py`.