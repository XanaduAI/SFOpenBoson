# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CV Hamiltonians
===============

Here, we provided several auxillary Hamiltonians for CV systems.

Gates
-----

This file contains the Hamiltonian representations of the CV
operations found in StrawberryFields.

All Hamiltonians in the file return a tuple containing the Hamiltonian
operator (either as a BosonOperator or a QuadOperator) and the time
propagation.

.. autosummary::
    displacement
    xdisplacement
    zdisplacement
    rotation
    squeezing
    quadratic_phase
    beamsplitter
    two_mode_squeezing
    controlled_addition
    controlled_phase
    cubic_phase
    kerr

Code details
------------
"""
from .gates import (displacement,
                    xdisplacement,
                    zdisplacement,
                    rotation,
                    squeezing,
                    quadratic_phase,
                    beamsplitter,
                    two_mode_squeezing,
                    controlled_addition,
                    controlled_phase,
                    cubic_phase,
                    kerr)

__all__ = ['displacement',
           'xdisplacement',
           'zdisplacement',
           'rotation',
           'squeezing',
           'quadratic_phase',
           'beamsplitter',
           'two_mode_squeezing',
           'controlled_addition',
           'controlled_phase',
           'cubic_phase',
           'kerr']
