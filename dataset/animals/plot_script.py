import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import os
from textwrap import wrap
from moviepy.editor import VideoClip
from pathlib import Path

def mplfig_to_npimage(fig):
    """ Converts a matplotlib figure to a RGB frame after updating the canvas"""
    #  only the Agg backend now supports the tostring_rgb function
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw() # update/draw the elements

    # get the width and the height to resize the matrix
    _,_,w,h = canvas.figure.bbox.bounds
    w, h = int(w), int(h)

    #  exports the canvas to a string buffer and then to a numpy nd.array
    buf = canvas.tostring_argb()
    image= np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return image[:, :, 1:4]


def get_general_skeleton_3d_motion(parents, joints, title, dataset, figsize=(7, 7), fps=120, radius=5, face_joints = [], fc = None):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=None)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    data = joints.copy().reshape(len(joints), -1, 3)
    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
    # elif dataset in ['truebones']: 
    #     data *= 0.2
    elif dataset in ['humanml', 'truebones', 'humanml_mat']:
        data *= 1.3  # scale for visualization
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    n_frames = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]


    def update(index):
        index = min(n_frames-1, int(index*fps))
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        for joint, parent in enumerate(parents[1:], start=1):
            ax.plot3D(data[index, [joint, parent], 0], data[index, [joint, parent], 1], data[index, [joint, parent], 2], color='red', solid_capstyle='round')
            if joint in face_joints:
                ax.scatter(data[index, joint, 0], data[index, joint, 1], data[index,joint, 2], color='blue', marker='o')
            if fc is not None and joint in fc[index]:
                ax.scatter(data[index, joint, 0], data[index, joint, 1], data[index,joint, 2], color='green', marker='o')
        
        plt.axis('off')
        ax.set_axis_off()
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        	
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        
        return mplfig_to_npimage(fig)

    ani = VideoClip(update)

    plt.close()
    return ani 
    
def save_sample(out_path, file_name, animation, fps, max_frames):
    sample_save_path = os.path.join(out_path, file_name)
    print(f'saving {file_name}')
    animation.duration = max_frames/fps
    animation.write_videofile(sample_save_path, fps=fps, threads=4, logger=None)
    animation.close()

def plot_general_skeleton_3d_motion(save_path, parents, joints, title, dataset="truebones", figsize=(7, 7), fps=120, radius=5, face_joints = [], fc = None):
    ani = get_general_skeleton_3d_motion(parents, joints, title, dataset, figsize, fps, radius, face_joints, fc)
    path = Path(save_path)
    out_dir = path.parent
    fname = path.name
    save_sample(out_dir, fname, ani, fps, len(joints))