# Text2Motion
<img src="assets/chicken_roars.gif" width="400"/>

A chicken roars

**Text2Motion** is a deep learning system that translates text descriptions into realistic 3D animations for any given mesh. It automates complex animation workflows with an intelligent, three-stage pipeline:

1. *Classification*: The system first analyzes the input 3D mesh to identify its key features and determine the appropriate armature (skeleton).

2. *Skinning*: It then binds the mesh to the armature, creating a "skinned" model ready for realistic deformation.

3. *Generation*: Finally, a powerful diffusion model generates fluid and nuanced motion based on your text prompt, bringing the model to life.

-----

## Getting Started

Follow these instructions to get the project up and running on your local machine.

1. **Setup environment**

    ```bash
      conda env create -f environment.yaml
      conda activate text2motion
    ```
2. **Download the dataset**

   This project uses the **Truebones Zoo dataset**, a collection of animated animal motions. You can get the dataset for free from the [Truebones Gumroad page](https://truebones.gumroad.com/l/skZMC).
   After downloading, extract the contents into a directory named `data` within the root of the project. The directory structure should look like this:

   ```
   text2motion/
   ├── data/
   │   └── Truebone_Z-OO/
   │       └── ... (dataset files)
   ├── train_classifier.py
   ├── train_diffusion.py
   └── ...
   ```

3. **Preprocess the dataset**

   To prepare the dataset for training, you must run two distinct preprocessing steps: one for the classification and skinning models, and a second one for the diffusion model.

   ### Preprocessing for Classification and Skinning

   This first step is essential for the classification and skinning models and requires **[Blender](https://www.blender.org/download/)**. You will need to run a script from your terminal that automates the necessary operations within Blender.

   Open a terminal and execute the following command:

   ```bash
   blender --background --python scripts/preprocessing_script.py
   ```

   -----

   ### Preprocessing for the Diffusion Model

   The second step prepares the data specifically for training the **diffusion model**. To correctly preprocess the data for the diffusion model, please refer to the detailed documentation and scripts provided in the **[Anytop repository](https://www.google.com/search?q=https.anytop2025.github.io/Anytop-page/)**. You should then put the `truebones_processed` folder in `data`.
-----

## Usage

### Training the Models

The complete training process involves three distinct stages that must be run **in order**. Each stage trains a separate model that is essential for the final animation pipeline.

1.  **Train the Classifier and skinning model**
    This model learns to analyze a 3D mesh and predict its skeletal structure (armature) in the correct positions with a weight for each vertex.

    ```bash
    python train_classifier_skinning.py
    ```

3.  **Train the Diffusion Model**
    This model learns to generate the actual motion sequence from a text prompt, which is then applied to the skinned model.

    ```bash
    python train_diffusion.py
    ```

-----

## Acknowledgments

* **[Truebones](https://truebones.gumroad.com/)** for providing the excellent and comprehensive Zoo dataset.
* **[Anytop](https://anytop2025.github.io/Anytop-page/)**, whose work provided the basis for our preprocessing scripts, visualization tools, and the core of our diffusion model.
* The authors of **[MDM](https://guytevet.github.io/mdm-page/)** for their insightful ideas that significantly influenced the architecture of our diffusion model.
* The **[RigNet](https://zhan-xu.github.io/rig-net/)** project for developing the `SkinNet` model, which we have adapted for our skinning process.
* The creators of **[PointNet++](https://arxiv.org/abs/1706.02413)** for the pioneering network architecture that underpins our classification and joint prediction models.

## License
This code is distributed under an [MIT LICENSE](LICENSE).
Note that our code depends on other libraries that have their own respective licenses that must also be followed.