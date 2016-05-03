from fipy.tools import dump
import fipy as fp


dx = dy = 0.025
nx = ny = 400


current_step = 9
uu, phase = dump.read('dump{0}.gz'.format(current_step))
u_viewer = fp.Viewer(uu)
phase_viewer = fp.Viewer(phase)
u_viewer.plot()
phase_viewer.plot
raw_input('stopped')
