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