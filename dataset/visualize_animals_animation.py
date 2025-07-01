"""
anim.npz ➜ GIF identiche al viewer BVH
pip install numpy matplotlib pillow
"""
import itertools
from pathlib import Path
import bvhio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ───────────────────‒ CONFIG ‒───────────────────
BVH_PATH   = Path("./dataset/Puppy_Run.bvh")   # << tuo file
DRAW_ENDS  = True                                # includi End Site / *Nub
FPS        = 30                                   # frame per secondo

# ───────────────────‒ LETTURA BVH ‒───────────────────
try:
    root = bvhio.read(str(BVH_PATH))             # bvhio ≥1.4
except AttributeError:
    root = bvhio.readAsHierarchy(str(BVH_PATH))

layout      = root.layout()                      # [(Joint, idx, depth)…]
joint2idx   = {j: i for i, (j, *_ ) in enumerate(layout)}
F, J        = len(root.Keyframes), len(layout)   # frames · joints

bones = []
for child, *_ in layout:
    if child.Parent and (DRAW_ENDS or not child.Name.endswith("Nub")):
        bones.append((joint2idx[child], joint2idx[child.Parent]))
bones = np.asarray(bones, np.int16)              # (B,2)

# world-coords per disegno veloce
xyz = np.empty((F, J, 3), np.float32)
for f in range(F):
    (root.loadPose if hasattr(root,"loadPose") else root.loadKeyframe)(f, True)
    for i, (j,*_) in enumerate(layout):
        xyz[f, i] = j.PositionWorld
xyz /= 100.0                                      # metri

hips = next(i for j, i in joint2idx.items() if j.Name == "Hips")
joints = xyz - xyz[:, hips:hips+1]

mi, ma = joints.min((0,1)), joints.max((0,1))
mid, R = (mi+ma)/2, (ma-mi).max()/2
lims   = [(mid[i]-R, mid[i]+R) for i in [0,2,1]]  # X,Z,Y

def make_scene(view, fname):
    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_box_aspect((1,1,1)); ax.axis("off"); ax.view_init(*view)

    scat  = ax.scatter([],[],[], c='k', s=10)
    lines = [ax.plot([],[],[], 'r-', lw=1)[0] for _ in bones]

    def init():
        scat._offsets3d = ([],[],[])
        for ln in lines: ln.set_data([],[]); ln.set_3d_properties([])
        return [scat]+lines

    def update(f):
        p = joints[f][:,[0,2,1]]                 # Y↔Z swap
        scat._offsets3d = (p[:,0], p[:,1], p[:,2])
        for (i,j), ln in zip(bones, lines):
            ln.set_data([p[i,0], p[j,0]],[p[i,1], p[j,1]])
            ln.set_3d_properties([p[i,2], p[j,2]])
        return [scat]+lines

    ani = FuncAnimation(fig, update, frames=F,
                        init_func=init, blit=True,
                        interval=1000/FPS)
    ani.save(fname, writer=PillowWriter(fps=FPS))
    print("✓ GIF generata:", fname)

make_scene((20,120), BVH_PATH.with_name(BVH_PATH.stem+"_front.gif"))
make_scene((90,-90), BVH_PATH.with_name(BVH_PATH.stem+"_top.gif"))
