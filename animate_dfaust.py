import time
import random
import argparse
from typing import Optional, Union

import numpy as np
import pyvista as pv
import torch
import logging

from dataset.dfaust import DFaustDataset

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DFaustAnimator:
    """
    Handles loading and visualizing animations from the DFaust dataset.
    """
    def __init__(self, num_to_show: int = 10, speed: float = 1.0) -> None:
        """
        Initializes the animator with the number of animations to show and speed.
        :param num_to_show: Number of animations to show
        :param speed: Speed multiplier for the animation (e.g., 2.0 = 2x faster, 0.5 = 2x slower)
        """
        self.num_to_show = num_to_show
        self.speed = max(speed, 0.01)  # Avoid zero or negative values
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> DFaustDataset:
        """
        Loads the DFaust dataset instance.

        :return: DFaustDataset object
        """
        try:
            dataset = DFaustDataset()
            if not dataset.samples:
                raise ValueError("No samples found in dataset.")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise RuntimeError(f"Failed to load dataset: {e}")

    def _convert_faces(self, faces_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Converts face array to PyVista format.

        :param faces_input: Array of face indices
        :return: Flattened face array for PyVista
        """
        if faces_input.ndim == 2 and faces_input.shape[1] == 3:
            return np.hstack((np.full((faces_input.shape[0], 1), 3), faces_input)).ravel()
        elif faces_input.ndim == 1:
            return faces_input
        return None

    def _prepare_vertices(self, verts_input: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Converts vertices to NumPy array.

        :param verts_input: Tensor or NumPy array of shape [frames, N, 3]
        :return: NumPy array of same shape
        """
        if isinstance(verts_input, torch.Tensor):
            return verts_input.cpu().numpy()
        elif isinstance(verts_input, np.ndarray):
            return verts_input
        else:
            raise TypeError("Unsupported vertex input type.")

    def _visualize(self, verts: np.ndarray, faces: Optional[np.ndarray], name: str) -> None:
        """
        Renders the animation in a PyVista window.

        :param verts: Array of vertices [frames, N, 3]
        :param faces: Face connectivity array or None
        :param name: Title for the window
        """
        num_frames = verts.shape[0]
        mesh = pv.PolyData(verts[0], faces=self._convert_faces(faces) if faces is not None else None)
        plotter = pv.Plotter(title=name)
        actor = plotter.add_mesh(mesh, cmap="viridis", smooth_shading=True)

        if faces is None and hasattr(actor, "actor"):
            actor.actor.property.point_size = 3

        plotter.camera_position = 'iso'
        plotter.camera.zoom(1.2)
        plotter.show(interactive_update=True, auto_close=False)

        try:
            for i in range(num_frames):
                mesh.points = verts[i]
                plotter.render()
                if plotter.render_window is None:
                    break
                time.sleep((1.0 / 30.0) / self.speed)
        except Exception as e:
            logger.error(f"[{name}] Animation error: {e}")
        finally:
            plotter.close()

    def run(self) -> None:
        """
        Starts the animation loop.
        """
        samples = list(self.dataset.samples)
        random.shuffle(samples)

        to_show = min(self.num_to_show, len(samples))
        logger.info(f"Showing {to_show} animation(s)")

        action_descriptions_dict = self.dataset.action_descriptions
        if not action_descriptions_dict:
            logger.warning("No action descriptions found.")

        for i in range(to_show):
            anim_id, anim_name = samples[i]
            logger.info(f"[{i + 1}/{to_show}] {anim_name}")
            description = action_descriptions_dict.get(anim_name, "No description available for this action.")
            logger.info(f"Description: {description}")
            try:
                verts, faces = self.dataset._load_animation(anim_id)
                verts_np = self._prepare_vertices(verts)
                self._visualize(verts_np, faces, anim_name)

                if i < to_show - 1:
                    choice = input("Enter = next | q = quit: ").strip().lower()
                    if choice.lower() == 'q':
                        break
            except Exception as e:
                logger.error(f"Error during animation '{anim_name}': {e}")
                choice = input("Enter = continue | q = quit: ").strip().lower()
                if choice.lower() == 'q':
                    break


def main():
    parser = argparse.ArgumentParser(description="Visualize DFaust animations")
    parser.add_argument('--num', type=int, default=10, help="Number of animations to show")
    parser.add_argument('--speed', type=float, default=2.0, help="Animation speed multiplier (e.g., 2.0 = 2x faster, 0.5 = 2x slower)")
    args = parser.parse_args()

    animator = DFaustAnimator(num_to_show=args.num, speed=args.speed)
    animator.run()


if __name__ == '__main__':
    main()
