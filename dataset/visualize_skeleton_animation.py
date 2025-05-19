import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

# Upload joints and center on root (index 0)
joints = np.load('./data/HumanML3D/new_joints/000000.npy')                        # (T,22,3)
joints = joints - joints[:, 0:1, :]                     # root→origin
joints_m = joints / 1000.0                              # mm→m

# Calculate global min/max on each axis
x_min, x_max = joints_m[:,:,0].min(), joints_m[:,:,0].max()
y_min, y_max = joints_m[:,:,1].min(), joints_m[:,:,1].max()
z_min, z_max = joints_m[:,:,2].min(), joints_m[:,:,2].max()

# Centred bounding box and radius
x_mid, y_mid, z_mid = (x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2
R = max(x_max-x_min, y_max-y_min, z_max-z_min) / 2
axes_limits = {
    'x': (x_mid-R, x_mid+R),
    'y': (y_mid-R, y_mid+R),
    'z': (z_mid-R, z_mid+R),
}

# Create list of bones
bones = []
for chain in t2m_kinematic_chain:
    for a, b in zip(chain[:-1], chain[1:]):
        bones.append((a, b))

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# Set axis limits
ax.set_xlim(*axes_limits['x'])
ax.set_ylim(*axes_limits['y'])
ax.set_zlim(*axes_limits['z'])

# Equal aspect ratio "manual"
ax.set_box_aspect((1,1,1))

# Initial view
ax.view_init(elev=20, azim=120)
ax.set_axis_off()

# Pre-create scatter and lines
scat = ax.scatter([], [], [], c='k', s=30)
lines = [ax.plot([], [], [], 'r-', linewidth=2)[0] for _ in bones]

def init():
    scat._offsets3d = ([], [], [])
    for ln in lines:
        ln.set_data([], [])
        ln.set_3d_properties([])
    return [scat] + lines

def update(t):
    # Switch Y e Z
    pts = joints_m[t][:, [0, 2, 1]]
    scat._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
    for (i,j), ln in zip(bones, lines):
        ln.set_data([pts[i,0], pts[j,0]],
                    [pts[i,1], pts[j,1]])
        ln.set_3d_properties([pts[i,2], pts[j,2]])
    return [scat] + lines

anim = FuncAnimation(fig, update,
                     frames=joints_m.shape[0],
                     init_func=init,
                     blit=True,
                     interval=1000/30)

anim.save('skeleton.gif', writer='imagemagick', fps=30)