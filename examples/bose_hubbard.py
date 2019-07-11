from openfermion.hamiltonians import bose_hubbard

import strawberryfields as sf
from strawberryfields.ops import *
from sfopenboson.ops import BoseHubbardPropagation

H = bose_hubbard(1, 2, 1, 1.5)

prog = sf.Program(2)
eng = sf.Engine("fock", backend_options={"cutoff_dim": 3})

with prog.context as q:
    Fock(2) | q[1]
    BoseHubbardPropagation(H, 1.086, 20) | q

state = eng.run(prog).state

print("Prob(2,0) = ", state.fock_prob([2,0]))
print("Prob(1,1) = ", state.fock_prob([1,1]))
print("Prob(0,2) = ", state.fock_prob([0,2]))
