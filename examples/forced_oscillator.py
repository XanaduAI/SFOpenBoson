import numpy as np

from matplotlib import pyplot as plt

from openfermion.ops import QuadOperator
from openfermion.utils import commutator, normal_ordered

import strawberryfields as sf
from strawberryfields.ops import *
from sfopenboson.ops import GaussianPropagation

# define the Hamiltonian
H = QuadOperator('q0 q0', 0.5) + QuadOperator('p0 p0', 0.5) - QuadOperator('q0', 2)

# create the engine
eng = sf.Engine("gaussian")

# set the time-steps
t_vals = np.arange(0, 6, 0.1)
results = np.zeros([2, len(t_vals)])

# evalutate the circuit at each time-step
for step, t in enumerate(t_vals):
    prog = sf.Program(1)

    with prog.context as q:
        Xgate(1) | q[0]
        Zgate(0.5) | q[0]
        GaussianPropagation(H, t) | q

    state = eng.run(prog).state
    eng.reset()
    results[:, step] = state.means()

# plot the results
plt.plot(*results)
plt.savefig('forced_oscillator.png')
