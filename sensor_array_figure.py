import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

r_cm = 3.3
L_cm = 7.9
S2 = np.sqrt(2)

sensors = np.array([
    [r_cm, 0, 0],
    [r_cm/S2, r_cm/S2, -L_cm],
    [0, r_cm, 0],
    [-r_cm/S2, r_cm/S2, -L_cm],
    [-r_cm, 0, 0],
    [-r_cm/S2, -r_cm/S2, -L_cm],
    [0, -r_cm, 0],
    [r_cm/S2, -r_cm/S2, -L_cm],
])

upper_idx = [0, 2, 4, 6]
lower_idx = [1, 3, 5, 7]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 원통형 가이드 원 그리기
theta = np.linspace(0, 2*np.pi, 100)
x_cyl = r_cm * np.cos(theta)
y_cyl = r_cm * np.sin(theta)

# 상단 원
ax.plot(x_cyl, y_cyl, np.zeros(100), color='#AAAAAA', lw=0.8, linestyle='--', alpha=0.6)
# 하단 원
ax.plot(x_cyl, y_cyl, -L_cm * np.ones(100), color='#AAAAAA', lw=0.8, linestyle='--', alpha=0.6)
# Vertical guide lines
for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
    ax.plot([r_cm*np.cos(angle)]*2, [r_cm*np.sin(angle)]*2, [0, -L_cm],
            color='#CCCCCC', lw=0.5, linestyle=':', alpha=0.4)

# Upper sensors (blue circles)
for i in upper_idx:
    ax.scatter(sensors[i, 0], sensors[i, 1], sensors[i, 2],
               c='#1565C0', s=200, zorder=5, edgecolors='white', linewidths=1.5)
    offset = [0.4, 0.4, 0.3]
    ax.text(sensors[i, 0] + offset[0], sensors[i, 1] + offset[1], sensors[i, 2] + offset[2],
            f'$S_{i}$', fontsize=11, fontweight='bold', color='#1565C0',
            ha='center', va='bottom')

# Lower sensors (red diamonds)
for i in lower_idx:
    ax.scatter(sensors[i, 0], sensors[i, 1], sensors[i, 2],
               c='#B71C1C', s=200, zorder=5, edgecolors='white', linewidths=1.5,
               marker='D')
    offset = [0.4, 0.4, -0.3]
    ax.text(sensors[i, 0] + offset[0], sensors[i, 1] + offset[1], sensors[i, 2] + offset[2],
            f'$S_{i}$', fontsize=11, fontweight='bold', color='#B71C1C',
            ha='center', va='top')

# Array center markers
ax.scatter(0, 0, 0, c='gray', s=50, marker='+', zorder=3, alpha=0.5)
ax.scatter(0, 0, -L_cm, c='gray', s=50, marker='+', zorder=3, alpha=0.5)

# Vertical spacing indicator and label
ax.plot([r_cm+0.8]*2, [0]*2, [0, -L_cm], color='#555555', lw=1.2,
        linestyle='-')
ax.plot([r_cm+0.5, r_cm+1.1], [0, 0], [0, 0], color='#555555', lw=1.2)
ax.plot([r_cm+0.5, r_cm+1.1], [0, 0], [-L_cm, -L_cm], color='#555555', lw=1.2)
ax.text(r_cm+1.3, 0, -L_cm/2, f'L = {L_cm} cm', fontsize=10,
        color='#333333', va='center')

# Radius indicator and label
ax.text(r_cm/2, -0.6, 0.2, f'r = {r_cm} cm', fontsize=10,
        color='#333333', ha='center')

# z-plane labels
ax.text(0, 0, -L_cm - 0.8, f'z = -{L_cm} cm', fontsize=9, color='#666666', ha='center')

# 45-degree stagger reference (between S0 and S1)
ax.plot([sensors[0,0], sensors[1,0]], [sensors[0,1], sensors[1,1]],
        [sensors[0,2], sensors[1,2]],
        color='#888888', lw=0.8, linestyle=':', alpha=0.5)

# Axis settings
ax.set_xlabel('X (cm)', fontsize=11, labelpad=8)
ax.set_ylabel('Y (cm)', fontsize=11, labelpad=8)
ax.set_zlabel('Z (cm)', fontsize=11, labelpad=8)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-10, 2)
ax.set_box_aspect([1, 1, 1.2])
ax.view_init(elev=20, azim=-45)
ax.grid(True, alpha=0.3)

# Legend
upper_patch = mpatches.Patch(color='#1565C0', label='Upper sensors ($S_0, S_2, S_4, S_6$) at $z = 0$')
lower_patch = mpatches.Patch(color='#B71C1C', label='Lower sensors ($S_1, S_3, S_5, S_7$) at $z = -L$')
ax.legend(handles=[upper_patch, lower_patch], loc='upper left',
          fontsize=9, framealpha=0.9)

plt.title('8-Channel Cylindrical Sensor Array Structure\n'
          r'($r = 3.3$ cm, $L = 7.9$ cm, 45° staggered arrangement)',
          fontsize=13, pad=15)

plt.tight_layout()
plt.savefig('sensor_array.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print('Saved: sensor_array.png')
