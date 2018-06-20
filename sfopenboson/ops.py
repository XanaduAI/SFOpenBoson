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
r"""
Operations
==========

This file contains the Strawberry Field quantum operations
that decompose the BosonOperator and QuadOperator from OpenFermion.

These operations are used directly in BlackBird code, complementing
existing operations.

For example:

.. code-block:: python

    eng, q = sf.Engine(3, hbar=2)

    H1 = BosonOperator('0^ 0')
    H2 = QuadOperator('q0 p0') + QuadOperator('p0 q0') - QuadOperator('p2 p2')

    with eng:
        GaussianPropagation(H1) | q[0]
        GaussianPropagation(H2, t=0.5, 'global') | q

    state = eng.run('gaussian')

The global argument indicates to Strawberry Fields that the Hamiltonian
should be applied to the entire register, with the operator indices
indicating the mode the operator acts on.

If 'global' is not provided, it is assumed that the Hamiltonian
should only be applied locally, to the qumodes specified on the right.
In this case, the number of operator indices must match the number of
qumodes the Hamiltonian is applied to.

To see the gates applied, simply run ``eng.print_applied()``:

>>> eng.print_applied()
Rgate(-1)   | (q[0])
Rgate(-0.3927)  | (q[2])
BSgate(-1.571, 0)   | (q[1], q[2])
Rgate(-3.142)   | (q[0])
Sgate(-2, 3.142)    | (q[0])
Sgate(-0.8814, 3.142)   | (q[1])
Rgate(-1.963)   | (q[1])
BSgate(-1.571, 0)   | (q[1], q[2])
Rgate(3.142)    | (q[0])


Blackbird quantum operations
----------------------------

.. autosummary::
    GaussianPropagation
    BoseHubbardPropagation

Code details
------------
"""
# pylint: disable=abstract-method,too-many-branches,too-many-arguments

import sys

import numpy as np
from scipy.linalg import expm, inv

from openfermion.ops import QuadOperator, BosonOperator
from openfermion.transforms import get_quad_operator, get_boson_operator
from openfermion.utils import is_hermitian, prune_unused_indices

from strawberryfields.ops import (BSgate,
                                  Decomposition,
                                  GaussianTransform,
                                  Kgate,
                                  Rgate,
                                  Xgate,
                                  Zgate)
from strawberryfields.engine import Engine as _Engine, Command
from strawberryfields.backends.shared_ops import sympmat

from .auxillary import trotter_layer, quadratic_coefficients


class GaussianPropagation(Decomposition):
    r"""Propagate the specified qumodes by a bosonic Gaussian Hamiltonian.

    A Gaussian Hamiltonian is any combination of quadratic operators
    that can be written in quadratic form:

    .. math:: H = \frac{1}{2}\mathbf{r}^T A\mathbf{r} + \mathbf{r}^T \mathbf{d}

    where:

    * :math:`A\in\mathbb{R}^{2N\times 2N}` is a symmetric matrix,
    * :math:`\mathbf{d}\in\mathbb{R}^{2N}` is a real vector, and
    * :math:`\mathbf{r} = (\x_1,\dots,\x_N,\p_1,\dots,\p_N)` is the vector
      of quadrature operators in :math:`xp`-ordering.

    This operation calculates the corresponding Gaussian symplectic
    transformation via the following relation:

    .. math:: S = e^{\Omega A t}

    where

    .. math::
        \Omega=\begin{bmatrix}0&I_N\\-I_N&0\end{bmatrix}\in\mathbb{R}^{2N\times 2N}

    is the symplectic matrix.

    Depending on whether the resulting symplectic transformation is passive
    (energy preserving) or active (non-energy preserving), the Clements or
    Bloch-Messiah decomposition in Strawberry Fields is then used to decompose
    the Hamiltonian into a set of CV gates.

    Args:
        operator (BosonOperator, QuadOperator): a bosonic Gaussian Hamiltonian
        t (float): the time propagation value. If not provided, default value is 1.
        mode (str): By default, ``mode='local'`` and the Hamiltonian is assumed to apply to only
            the applied qumodes (q[i], q[j],...). For instance, a_0 applies to q[i], a_1 applies to q[j].
            If instead ``mode='global'``, the Hamiltonian is instead applied to the entire register,
            i.e., a_0 applies to q[0], applies to q[1], etc.
        hbar (float): the value of :math:`\hbar` used in the definition of the :math:`\x`
            and :math:`\p` quadrature operators. Note that if used inside of an engine
            context, the hbar value of the engine will override this keyword argument.
    """
    ns = None
    def __init__(self, operator, t=1, mode='local', hbar=None):
        super().__init__([t, operator])

        try:
            # pylint: disable=protected-access
            self.hbar = _Engine._current_context.hbar
        except AttributeError:
            if hbar is None:
                raise ValueError("Either specify the hbar keyword argument, "
                                 "or use this operator inside an engine context.")
            else:
                self.hbar = hbar

        if not is_hermitian(operator):
            raise ValueError("Hamiltonian must be Hermitian.")

        if mode == 'local':
            quad_operator = prune_unused_indices(operator)
        elif mode == 'global':
            quad_operator = operator

        if isinstance(quad_operator, BosonOperator):
            quad_operator = get_quad_operator(quad_operator, hbar=self.hbar)

        A, d = quadratic_coefficients(quad_operator)

        if mode == 'local':
            self.ns = A.shape[0]//2
        elif mode == 'global':
            # pylint: disable=protected-access
            self.ns = _Engine._current_context.num_subsystems
            if A.shape[0] < 2*self.ns:
                # expand the quadratic coefficient matrix to take
                # into account the extra modes
                A_n = A.shape[0]//2
                tmp = np.zeros([2*self.ns, 2*self.ns])

                tmp[:A_n, :A_n] = A[:A_n, :A_n]
                tmp[:A_n, self.ns:self.ns+A_n] = A[:A_n, A_n:]
                tmp[self.ns:self.ns+A_n, :A_n] = A[A_n:, :A_n]
                tmp[self.ns:self.ns+A_n, self.ns:self.ns+A_n] = A[A_n:, A_n:]

                A = tmp

        self.S = expm(sympmat(self.ns) @ A * t)

        self.disp = False
        if not np.all(d == 0.):
            self.disp = True
            if np.all(A == 0.):
                self.d = d*t
            else:
                if np.linalg.cond(A) >= 1/sys.float_info.epsilon:
                    # the matrix is singular, add a small epsilon
                    eps = 1e-9
                    epsI = eps * np.identity(2*self.ns)
                    s = inv(A+epsI) @ d
                    tmp = (np.identity(2*self.ns) \
                        - expm(sympmat(self.ns) @ (A+epsI) * t).T) @ s / eps
                else:
                    s = inv(A) @ d
                    tmp = s - self.S.T @ s

                self.d = np.zeros([2*self.ns])
                self.d[self.ns:] = tmp[:self.ns]
                self.d[:self.ns] = tmp[self.ns:]

    def decompose(self, reg):
        """Return the decomposed commands"""
        cmds = []
        cmds += [Command(GaussianTransform(self.S, hbar=self.hbar), reg, decomp=True)]
        if self.disp:
            cmds += [Command(Xgate(x), reg, decomp=True) for x in self.d[:self.ns] if x != 0.]
            cmds += [Command(Zgate(z), reg, decomp=True) for z in self.d[self.ns:] if z != 0.]
        return cmds


class BoseHubbardPropagation(Decomposition):
    r"""Propagate the specified qumodes by a Bose-Hubbard Hamiltonian.

    The Bose-Hubbard Hamiltonian has the form

    .. math::
        H = -J\sum_{i=1}^N\sum_{j=1}^N A_{ij} \ad_i\a_j
            + \frac{1}{2}U\sum_{i=1}^N \a_i^\dagger \a_i (\ad_i \a_i - 1)
            - \mu \sum_{i=1}^N \ad_i \a_i
            + V \sum_{i=1}^N\sum_{j=1}^N A_{ij} \ad_i \a_i \ad_j \a_j.

    where:

    * :math:`A` is a real symmetric matrix of ones and zeros defining the adjacency of
      each pairwise combination of nodes :math:`(i,j)` in the :math:`N`-node system,
    * :math:`J` represents the transfer integral or hopping term of the boson between nodes,
    * :math:`U` is the on-site interaction potential,
    * :math:`\mu` is the chemical potential,
    * :math:`V` is the dipole-dipole or nearest neighbour interaction term.

    BoseHubbard Hamiltonians can be generated using the BosonOperator manually, or
    on a (periodic/non-peridic) two-dimensional lattice via the function
    ``openfermion.hamiltonians.bose_hubbard``
    (see the `OpenFermion documentation <http://openfermion.readthedocs.io/en/latest/openfermion.html#openfermion.hamiltonians.bose_hubbard>`_).

    In Strawberry Fields, the Bose-Hubbard propagation is performed by applying the
    Lie-product formula, and decomposing the unitary operations into a combination
    of beamsplitters, Kerr gates, and phase-space rotations.

    .. note:: Nearest-neighbour interactions (:math:`V\neq 0`) are not currently supported.

    Args:
        operator (BosonOperator, QuadOperator): a bosonic Gaussian Hamiltonian
        t (float): the time propagation value. If not provided, default value is 1.
        k (int): the number of products in the truncated Lie product formula.
        mode (str): By default, ``mode='local'`` and the Hamiltonian is assumed to apply to only
            the applied qumodes (q[i], q[j],...). For instance, a_0 applies to q[i], a_1 applies to q[j].
            If instead ``mode='global'``, the Hamiltonian is instead applied to the entire register,
            i.e., a_0 applies to q[0], applies to q[1], etc.
        hbar (float): the value of :math:`\hbar` used in the definition of the :math:`\x`
            and :math:`\p` quadrature operators. Note that if used inside of an engine
            context, the hbar value of the engine will override this keyword argument.
    """
    ns = None
    def __init__(self, operator, t=1, k=20, mode='local', hbar=None):
        super().__init__([t, operator])

        try:
            # pylint: disable=protected-access
            self.hbar = _Engine._current_context.hbar
        except AttributeError:
            if hbar is None:
                raise ValueError("Either specify the hbar keyword argument, "
                                 "or use this operator inside an engine context.")
            else:
                self.hbar = hbar

        if not is_hermitian(operator):
            raise ValueError("Hamiltonian must be Hermitian.")

        if (not isinstance(k, int)) or k <= 0:
            raise ValueError("Argument k must be a postive integer.")

        if mode == 'local':
            boson_operator = prune_unused_indices(operator)
        elif mode == 'global':
            boson_operator = operator

        if isinstance(boson_operator, QuadOperator):
            boson_operator = get_boson_operator(boson_operator, hbar=self.hbar)

        self.layer = trotter_layer(boson_operator, t, k)
        self.num_layers = k

        num_modes = max([op[0] for term in operator.terms for op in term])+1

        if mode == 'local':
            self.ns = num_modes
        elif mode == 'global':
            # pylint: disable=protected-access
            self.ns = _Engine._current_context.num_subsystems

    def decompose(self, reg):
        # make BS gate
        theta = self.layer['BS'][0]
        phi = self.layer['BS'][1]
        BS = BSgate(theta, phi)

        # make Kerr gate
        K = Kgate(self.layer['K'][0])

        # make rotation gate
        R = Rgate(self.layer['R'][0])

        cmds = []

        for i in range(self.num_layers): #pylint: disable=unused-variable
            for q0, q1 in self.layer['BS'][2]:
                cmds.append(Command(BS, (reg[q0], reg[q1])))

            for mode in self.layer['K'][1]:
                cmds.append(Command(K, reg[mode]))

            for mode in self.layer['R'][1]:
                cmds.append(Command(R, reg[mode]))

        return cmds
