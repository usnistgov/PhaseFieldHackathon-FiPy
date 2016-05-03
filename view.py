from fipy.tools import dump
import fipy as fp
import numpy as np
import sys

dx = dy = 0.25
nx = ny = 400

mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

def get_vars(current_step):
    data = np.load('data/dump{0}.npz'.format(current_step))
    uu = fp.CellVariable(mesh=mesh, value=data['uu'])
    phase = fp.CellVariable(mesh=mesh, value=data['phase'])
    return uu, phase, data['elapsed_time']

current_step = sys.argv[1]
uu, phase, elapsed_time = get_vars(current_step)

u_viewer = fp.Viewer(uu)
phase_viewer = fp.Viewer(phase)
u_viewer.plot()
phase_viewer.plot()
print 'elapsed_time',elapsed_time

counts = []
times = []
for step in  np.arange(10, int(current_step) + 1, 10):
    uu, phase, elapsed_time = get_vars(step)
    counts.append((np.array(phase) > 0).sum())
    times.append(elapsed_time)

area0 = counts[0] * dx * dy
r0 = np.sqrt(area0 / np.pi)
area1 = counts[-1] * dx * dy
r1 = np.sqrt(area1 / np.pi)
t0 = times[0]
t1 = times[-1]
print 'exapansion rate:',(r1 - r0) / (t1 - t0)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(times, counts)
plt.show()
raw_input('stopped')
