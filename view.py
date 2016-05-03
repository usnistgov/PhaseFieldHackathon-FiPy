from fipy.tools import dump
import fipy as fp
import numpy as np

dx = dy = 0.025
nx = ny = 400

mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

current_step = 2
data = np.load('dump{0}.npz'.format(current_step))
uu = fp.CellVariable(mesh=mesh, value=data['uu'])
phase = fp.CellVariable(mesh=mesh, value=data['phase'])
u_viewer = fp.Viewer(uu)
phase_viewer = fp.Viewer(phase)
u_viewer.plot()
phase_viewer.plot
raw_input('stopped')
