
# coding: utf-8

# In[257]:



import fipy as fp
import fipy.tools.numerix as numerix
from fipy.tools import dump

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
delta = 0.05


# In[374]:

phase = fp.CellVariable(mesh=mesh, hasOld=True)
uu = fp.CellVariable(mesh=mesh, hasOld=True)
uu.constrain(-delta, mesh.exteriorFaces)
dt = fp.Variable(1.0)


# In[375]:

def initialize():
    phase[:] = -1.0
    x, y = mesh.cellCenters
    radius = 2.0
    center = (nx * dx / 2., ny * dy / 2.)
    mask = (x - center[0])**2 + (y - center[1])**2 < radius**2
    phase.setValue(1., where=mask)
    uu[:] = -delta
    dt = fp.Variable(0.1)
initialize()


# In[376]:

def make_tau(phase_):
    theta_cell = numerix.arctan2(phase_.grad[1], phase_.grad[0])
    a_cell = 1 + epsilon_m * numerix.cos(mm * theta_cell + theta_0)
    return tau_0 * a_cell**2

tau = make_tau(phase)
tau_old = make_tau(phase.old)


# In[377]:

source_explicit = 2 * phase**3 - lamda * uu * (1 - phase**2) * (1 + 3 * phase**2)
source_implicit = (1 - 3 * phase**2) + 4 * lamda * uu * phase * (1 - phase**2)


# In[378]:

theta = numerix.arctan2(phase.faceGrad[1], phase.faceGrad[0])
W = W_0 * (1 + epsilon_m * numerix.cos(mm * theta + theta_0))

W_theta = - W_0 * mm * epsilon_m * numerix.sin(mm * theta + theta_0)

I0 = fp.Variable(value=((1,0), (0,1)))
I1 = fp.Variable(value=((0,-1), (1,0)))

Dphase = W**2 * I0 + W * W_theta * I1


# In[379]:

heat_eqn = fp.TransientTerm() == fp.DiffusionTerm(DD) + (phase - phase.old) / dt / 2.

phase_eqn = fp.TransientTerm(tau) == fp.DiffusionTerm(Dphase) + source_explicit + fp.ImplicitSourceTerm(source_implicit) #+ fp.ImplicitSourceTerm((tau - tau_old) / dt)


# In[380]:

print max(tau), min(tau)


# In[360]:

phase_viewer = fp.Viewer(phase)
phase_viewer.plot()


# In[361]:

print tau_old


# In[367]:



# In[ ]:

initialize()
dt.setValue(0.01)
total_steps = 20
sweeps = 5
tolerance = 1e-1
from fipy.solvers.pysparse import LinearLUSolver as Solver
solver_heat = Solver()
solver_phase = Solver()

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
        current_step += 1
        dt.setValue(dt.value * 1.1)
        dump.write((uu, phase), 'dump{0}.gz'.format(current_step))
    else:
        dt.setValue(dt.value * 0.8)
        uu[:] == uu.old
        phase[:] = phase.old



# In[371]:

phase_viewer.plot()


# In[335]:

print phase


# In[ ]:
