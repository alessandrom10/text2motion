"""
BVH  ➜  anim.npz   (+ GIF di debug)
dipendenze: pip install --upgrade bvhio numpy matplotlib pillow
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import bvhio

# ───────────────────‒ CONFIG ‒───────────────────
BVH_PATH   = Path("./dataset/__TravelWeb.bvh")   # << tuo file
OUT_NPZ    = BVH_PATH.with_suffix(".npz")        # anim.npz
FPS_GIF    = 30
DRAW_ENDS  = True                                # includi End Site / *Nub

# ───────────────────‒ LETTURA BVH ‒───────────────────
try:
    root = bvhio.read(str(BVH_PATH))             # bvhio ≥1.4
except AttributeError:
    root = bvhio.readAsHierarchy(str(BVH_PATH))

layout      = root.layout()                      # [(Joint, idx, depth)…]
joint2idx   = {j: i for i, (j, *_ ) in enumerate(layout)}
F, J        = len(root.Keyframes), len(layout)   # frames · joints

# ───────────────────‒ BONES padre→figlio (B×2) ‒───────────────────
bones = []
for child, *_ in layout:
    if child.Parent and (DRAW_ENDS or not child.Name.endswith("Nub")):
        bones.append((joint2idx[child], joint2idx[child.Parent]))
bones = np.asarray(bones, np.int16)              # (B,2)

# ──────────────── MOTION (F × J × 6) ────────────────
motion = np.empty((F, J, 6), np.float32)           # [:,:3] posL, [:,3:] rotL

for f in range(F):
    # carico i valori locali del frame f
    (root.loadPose if hasattr(root, "loadPose") else root.loadKeyframe)(f, False)

    for i, (j, *_ ) in enumerate(layout):

        # ---------- 3 componenti di traslazione locale ----------
        if hasattr(j, "Offset"):                   # bvhio "recente"
            pos_cm = j.Offset + j.Position         # cm
        else:                                      # bvhio "lite": Position già completa
            pos_cm = j.Position                    # cm
        motion[f, i, :3] = pos_cm / 100.0          # → metri

        # ---------- 3 componenti di rotazione locale ----------
        rot_attr = getattr(j, "Rotation", None)

        if rot_attr is not None:                  # build “aggregata”
            rot = np.asarray(rot_attr, np.float32)
            if rot.size == 4:                     # quaternion → Euler Z-X-Y
                w, x, y, z = rot
                # formule standard di conversione (angoli in gradi)
                t0 = +2*(w*z + x*y)
                t1 = +1 - 2*(y*y + z*z)
                rz = np.degrees(np.arctan2(t0, t1))

                t2 = +2*(w*x - y*z)
                t2 = np.clip(t2, -1.0, 1.0)
                rx = np.degrees(np.arcsin(t2))

                t3 = +2*(w*y + x*z)
                t4 = +1 - 2*(x*x + y*y)
                ry = np.degrees(np.arctan2(t3, t4))

                motion[f, i, 3:6] = (rz, rx, ry)
            elif rot.size == 3:                   # già Z-X-Y
                motion[f, i, 3:6] = rot
            else:                                 # vettore vuoto ➜ (0,0,0)
                motion[f, i, 3:6] = 0.0
        else:                                     # build “classica” RotationZ/… 
            motion[f, i, 3] = getattr(j, "RotationZ", 0.0)
            motion[f, i, 4] = getattr(j, "RotationX", 0.0)
            motion[f, i, 5] = getattr(j, "RotationY", 0.0)


print(f"motion shape  : {motion.shape}  (F,J,6)")
print(f"bones  shape  : {bones.shape}   (B,2)")

# ───────────────────‒ SALVA IN UNICO NPZ ‒───────────────────
#np.savez(OUT_NPZ,
#         bones   = bones,
#         motion  = motion.astype(np.float32))
#print("✓ salvato", OUT_NPZ.name)

# ───────────────────‒ DEBUG opzionale: GIF scheletro ‒───────────────────
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
                        interval=1000/FPS_GIF)
    ani.save(fname, writer=PillowWriter(fps=FPS_GIF))
    print("✓ GIF generata:", fname)

make_scene((20,120), BVH_PATH.with_name(BVH_PATH.stem+"_front.gif"))
make_scene((90,-90), BVH_PATH.with_name(BVH_PATH.stem+"_top.gif"))
