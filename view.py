from fipy import CellVariable, Grid2D, Variable, Viewer
from fipy.tools import dump
import numpy as np
import sys

dx = dy = 0.25
nx = ny = 400

mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

def get_vars(current_step):
    data = np.load('data-test/dump{0}.npz'.format(current_step))
    uu = CellVariable(mesh=mesh, value=data['uu'])
    phase = CellVariable(mesh=mesh, value=data['phase'])
    return uu, phase, data['elapsed_time']

current_step = sys.argv[1]
uu, phase, elapsed_time = get_vars(current_step)

u_viewer = Viewer(uu,title=r'$u$ at $t=$%.3f'%(elapsed_time))
phase_viewer = Viewer(phase,title=r'$\phi$ at $t=$%.3f'%elapsed_time)
u_viewer.plot()
phase_viewer.plot()
print 'elapsed_time',elapsed_time
D = 10

v = np.array(-(uu.faceGrad * Variable((1,0))).divergence * dx * D)
v_line = v.reshape((nx,ny))[nx / 2]

counts = []
counts1d = []
times = []
crossings = []

for step in  np.arange(10, int(current_step) + 1, 10):
    uu, phase, elapsed_time = get_vars(step)
    counts.append((np.array(phase) > 0).sum())
    count1d = (np.array(phase).reshape((nx,ny))[nx / 2] > 0).sum()
    counts1d.append(count1d)
    times.append(elapsed_time)
    # Estimate 0-level crossing (dendrite tip location)
    p_line = np.array(phase).reshape((nx,ny))[nx / 2]
    i = nx/2
    while p_line[i] > 0.:
        i += 1
    x_l = dx * (i-1)
    x_r = dx * i
    phi_l = p_line[i-1]
    phi_r = p_line[i]
    crossings.append(x_l + (-phi_l/(phi_r - phi_l)))

area0 = counts[0] * dx * dy
r0 = np.sqrt(area0 / np.pi)
area1 = counts[-1] * dx * dy
r1 = np.sqrt(area1 / np.pi)
t0 = times[-10]
t1 = times[-1]
r0_ = dx * counts1d[-10] / 2
r1_ = dx * counts1d[-1] / 2

print 'exapansion rate:',(r1 - r0) / (t1 - t0)
print 'tip velocity:',(r1_ - r0_) / (t1 - t0)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(v_line)
plt.xlabel('position')
plt.ylabel('$v$')
plt.title('Growth Velocity')
plt.savefig('growth_velocity.png', dpi=400, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(times, counts)
plt.xlabel('$t$')
plt.ylabel('$A$')
plt.savefig('solid_area.png', dpi=400, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(times,crossings)
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.title('Tip Position')
plt.savefig('tip_position.png', dpi=400, bbox_inches='tight')
plt.close()

raw_input('stopped')
