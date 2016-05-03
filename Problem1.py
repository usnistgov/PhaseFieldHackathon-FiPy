
# coding: utf-8

# In[257]:



import fipy as fp
import fipy.tools.numerix as numerix
from fipy.tools import dump
import numpy as np

# In[372]:

nx = 400
ny = 400
dx = 0.25
dy = 0.25
mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

# In[373]:

delta = 0.0
mm = 4.
epsilon_m = 0.025
theta_0 = 0.0
tau_0 = 1.
DD = 10.
W_0 = 1.
lamda = DD * tau_0 / 0.6267 / W_0**2
delta = 0.5


# In[374]:

phase = fp.CellVariable(mesh=mesh, hasOld=True)
uu = fp.CellVariable(mesh=mesh, hasOld=True)
uu.constrain(-delta, mesh.exteriorFaces)
dt = fp.Variable(0.01)


# In[375]:

def initialize():
    phase[:] = -1.0
    x, y = mesh.cellCenters
    radius = 4.0
    center = (nx * dx / 2., ny * dy / 2.)
    mask = (x - center[0])**2 + (y - center[1])**2 < radius**2
    phase.setValue(1., where=mask)
    uu[:] = -delta

initialize()


# In[376]:

def make_tau(phase_):
    theta_cell = numerix.arctan2(phase_.grad[1], phase_.grad[0])
    a_cell = 1 + epsilon_m * numerix.cos(mm * theta_cell + theta_0)
    return tau_0 * a_cell**2

tau = make_tau(phase)
tau_old = make_tau(phase.old)

# In[377]:

source = (phase - lamda * uu * (1 - phase**2)) * (1 - phase**2)


# In[378]:

theta = numerix.arctan2(phase.faceGrad[1], phase.faceGrad[0])
W = W_0 * (1 + epsilon_m * numerix.cos(mm * theta - theta_0))

W_theta = - W_0 * mm * epsilon_m * numerix.sin(mm * theta - theta_0)

I0 = fp.Variable(value=((1,0), (0,1)))
I1 = fp.Variable(value=((0,-1), (1,0)))

Dphase = W**2 * I0 + W * W_theta * I1


# In[379]:

heat_eqn = fp.TransientTerm() == fp.DiffusionTerm(DD) + (phase - phase.old) / dt / 2.

phase_eqn = fp.TransientTerm(tau) == fp.DiffusionTerm(Dphase) + source


# In[380]:

print max(tau), min(tau)





print tau_old


# In[367]:



# In[ ]:

initialize()
dt.setValue(0.01)
total_steps = 20000
sweeps = 5
tolerance = 1e-1
from fipy.solvers.pysparse import LinearLUSolver as Solver
solver_heat = Solver()
solver_phase = Solver()
elapsed_time = 0.0
current_step = 0

while current_step < total_steps:
    uu.updateOld()
    phase.updateOld()

    res_heat0 = heat_eqn.sweep(uu, dt=dt.value, solver=solver_heat)
    res_phase0 = phase_eqn.sweep(phase, dt=dt.value, solver=solver_phase)

    for sweep in range(sweeps):
        res_heat = heat_eqn.sweep(uu, dt=dt.value, solver=solver_heat)
        res_phase = phase_eqn.sweep(phase, dt=dt.value, solver=solver_phase)


    print
    print 'dt',dt.value
    print 'current_step',current_step
    print 'res_heat',res_heat0, res_heat
    print 'res_phase',res_phase0, res_phase
    if (res_heat < res_heat0 * tolerance) and (res_phase < res_phase0 * tolerance):
        elapsed_time += dt.value
        current_step += 1
        dt.setValue(dt.value * 1.1)
        if current_step % 10 == 0:
            np.savez_compressed('data-test/dump{0}.npz'.format(current_step), uu=np.array(uu), phase=np.array(phase), elapsed_time=np.array(elapsed_time))
    else:
        dt.setValue(dt.value * 0.8)
        uu[:] == uu.old
        phase[:] = phase.old



# In[371]:



# In[335]:

print phase


# In[ ]:
