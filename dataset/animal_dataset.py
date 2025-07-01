from pathlib import Path
import glm
import numpy as np
import bvhio

# ─────────── SETTING ───────────
BVH_ROOT = Path("./data")
OUT_ROOT = Path("./exports_npz")
DRAW_ENDS = True                             
OUT_ROOT.mkdir(parents=True, exist_ok=True)  

# ─────────── UTILITY FUNCTIONS ───────────
def output_path_for(bvh_path: Path) -> Path:
    """Apply renaming rules to get the target .npz file."""
    rel = bvh_path.relative_to(BVH_ROOT)
    if rel.parent == Path("."):
        return OUT_ROOT / rel.with_suffix(".npz")

    folder = rel.parent.name 
    stem   = rel.stem 
    if stem.startswith(folder): 
        new_name = f"{stem}.npz"
    else:
        base = stem.lstrip("_")
        new_name = f"{folder}__{base}.npz"

    return OUT_ROOT / rel.parent / new_name

def bvh2npz(bvh_file: Path, out_file: Path):
    """Convert a single .bvh file to .npz (same logic as your original script)."""
    try:
        root = bvhio.read(str(bvh_file))
    except AttributeError:
        root = bvhio.readAsHierarchy(str(bvh_file))

    layout    = root.layout()
    joint2idx = {j: i for i, (j, *_ ) in enumerate(layout)}
    F, J      = len(root.Keyframes), len(layout)

    bones = [
        (joint2idx[ch], joint2idx[ch.Parent])
        for ch, *_ in layout
        if ch.Parent and (DRAW_ENDS or not ch.Name.endswith("Nub"))
    ]
    bones = np.asarray(bones, np.int16)

    # motion (F × J × 6)
    motion = np.empty((F, J, 6), np.float32)
    for f in range(F):
        root.loadKeyframe(f, recursive=True)
        for i, (j, *_ ) in enumerate(layout):
            off_cm = getattr(j, "RestPose", None)
            off_cm = off_cm.Position if off_cm else glm.vec3(0)
            pos_cm = off_cm + j.Position
            motion[f, i, :3] = np.asarray(pos_cm) / 100.0

            
            rot = getattr(j, "Rotation", None)
            if rot is not None:                      
                rot = np.asarray(rot, np.float32)
                if rot.size == 4:                   
                    w, x, y, z = rot
                    t0, t1     =  2*(w*z + x*y),  1 - 2*(y*y + z*z)
                    rz         = np.degrees(np.arctan2(t0, t1))
                    t2         = np.clip(2*(w*x - y*z), -1.0, 1.0)
                    rx         = np.degrees(np.arcsin(t2))
                    t3, t4     =  2*(w*y + x*z),  1 - 2*(x*x + y*y)
                    ry         = np.degrees(np.arctan2(t3, t4))
                    motion[f, i, 3:6] = (rz, rx, ry)
                elif rot.size == 3:                  
                    motion[f, i, 3:6] = rot
                else:
                    motion[f, i, 3:6] = 0.0
            else:                                    
                motion[f, i, 3] = getattr(j, "RotationZ", 0.0)
                motion[f, i, 4] = getattr(j, "RotationX", 0.0)
                motion[f, i, 5] = getattr(j, "RotationY", 0.0)

    # save
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_file, bones=bones, motion=motion.astype(np.float32))
    print(f"✓ {bvh_file.relative_to(BVH_ROOT)} → {out_file.relative_to(OUT_ROOT)}")

# ─────────── SCANNING & CONVERSION ───────────
bvh_files = list(BVH_ROOT.rglob("*.bvh"))
if not bvh_files:
    raise FileNotFoundError(f"No .bvh found in {BVH_ROOT.resolve()}")

for bf in bvh_files:
    bvh2npz(bf, output_path_for(bf))
