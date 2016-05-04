
# coding: utf-8

from fipy import CellVariable, DiffusionTerm, Grid2D, parallelComm, TransientTerm, Variable
from fipy.tools import dump, numerix
import numpy as np

# Numerical parameters
nx = ny = 400         # domain size
dx = dy = 0.25        # mesh resolution
dt = Variable(0.1) # initial timestep

# Physical parameters
mm = 4.               # anisotropic symmetry
epsilon_m = 0.025     # degree of anisotropy
theta_0 = 0.0         # tilt w.r.t. x-axis
tau_0 = 1.            # numerical mobility
DD = 10.              # thermal diffusivity
W_0 = 1.              # isotropic well height
lamda = DD * tau_0 / 0.6267 / W_0**2
delta = 0.05          # undercooling

# Mesh and field variables
mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
phase = CellVariable(mesh=mesh, hasOld=True)
uu = CellVariable(mesh=mesh, hasOld=True)
uu.constrain(-delta, mesh.exteriorFaces)


def initialize():
    phase[:] = -1.0
    x, y = mesh.cellCenters
    radius = 4.0 # Initial r=1 collapses due to Gibbs-Thomson, r=2 slumps to phi=0.6, r=4 seems OK.
    center = (nx * dx / 2., ny * dy / 2.)
    mask = (x - center[0])**2 + (y - center[1])**2 < radius**2
    phase.setValue(1., where=mask)
    uu[:] = -delta

initialize()


def make_tau(phase_):
    theta_cell = numerix.arctan2(phase_.grad[1], phase_.grad[0])
    a_cell = 1 + epsilon_m * numerix.cos(mm * theta_cell + theta_0)
    return tau_0 * a_cell**2

tau = make_tau(phase)
tau_old = make_tau(phase.old)

source = (phase - lamda * uu * (1 - phase**2)) * (1 - phase**2)


theta = numerix.arctan2(phase.faceGrad[1], phase.faceGrad[0])
W = W_0 * (1 + epsilon_m * numerix.cos(mm * theta - theta_0))

W_theta = - W_0 * mm * epsilon_m * numerix.sin(mm * theta - theta_0)

# Build up the diffusivity matrix
I0 = Variable(value=((1,0), (0,1)))
I1 = Variable(value=((0,-1), (1,0)))
Dphase = W**2 * I0 + W * W_theta * I1


heat_eqn = TransientTerm() == DiffusionTerm(DD) + (phase - phase.old) / dt / 2.

phase_eqn = TransientTerm(tau) == DiffusionTerm(Dphase) + source

initialize()

solid_area = (np.array(phase.globalValue)>0).sum()*dx*dy # initial size of solid nucleus
if parallelComm.procID==0:
    print 'solid area', solid_area


total_steps = 20000
sweeps = 3
tolerance = 0.1

# Serial:
#from fipy.solvers.pysparse import LinearLUSolver as Solver
#solver_heat = Solver()
#solver_phase = Solver()

# Parallel: mpirun with --trilinos for proper meshing, etc.
from fipy.solvers.trilinos import LinearGMRESSolver as Solver
solver_heat = Solver(precon=None)
solver_phase = Solver(precon=None)

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

    solid_area = (np.array(phase.globalValue)>0).sum()*dx*dy # initial size of solid nucleus

    if parallelComm.procID==0:
        print
        print 'dt',dt.value
        print 'current_step',current_step
        print 'res_heat',res_heat0, res_heat
        print 'res_phase',res_phase0, res_phase
        print 'solid area', solid_area
    if (res_heat < res_heat0 * tolerance) and (res_phase < res_phase0 * tolerance):
        elapsed_time += dt.value
        current_step += 1
        dt.setValue(dt.value * 1.1)
        if current_step % 10 == 0:
            glu = uu.globalValue
            glp = phase.globalValue
            if parallelComm.procID==0:
                np.savez_compressed('data-test/dump{0}.npz'.format(current_step), uu=np.array(glu), phase=np.array(glp), elapsed_time=np.array(elapsed_time))
    else:
        dt.setValue(dt.value * 0.8)
        uu[:] == uu.old
        phase[:] = phase.old
