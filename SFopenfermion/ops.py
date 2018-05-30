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
In this case, number of operator indices must match the number of
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


Summary
-------

.. autosummary::
    quadratic_coefficients
    GaussianPropagation

Code details
------------
"""

import sys

import numpy as np
from scipy.linalg import expm, inv

from strawberryfields.ops import (Xgate, Zgate,
    Decomposition, GaussianTransform)
from strawberryfields.engine import Engine as _Engine, Command
from strawberryfields.backends.shared_ops import sympmat, changebasis

from openfermion.ops import QuadOperator, BosonOperator
from openfermion.transforms import get_quad_operator
from openfermion.utils import is_hermitian, prune_unused_indices


def quadratic_coefficients(operator):
    r"""Return the quadratic coefficient matrix representing a Gaussian Hamiltonian.

    A Gaussian Hamiltonian is any combination of quadratic operators
    that can be written in quadratic form:

    .. math:: H = \frac{1}{2}\mathbf{r}A\mathbf{r} + \mathbf{r}^T \mathbf{d}

    where :math:`A\in\mathbb{R}^{2N\times 2N}` is a symmetric matrix,
    :math:`\mathbf{d}\in\mathbb{R}^{2N}` is a real vector, and
    :math:`\mathbf{r} = (\x_1,\dots,\x_N,\p_1,\dots,\p_N)` is the vector
    of means in :math:`xp`-ordering.

    This function accepts a bosonic Gaussian Hamiltonian, and returns the
    matrix :math:`A` and vector :math:`\mathbf{d}` representing the
    quadratic and linear coefficients.

    Args:
        operator (QuadOperator): a bosonic Gaussian Hamiltonian
    Returns:
        tuple(A, d): a tuple contains a 2Nx2N real symmetric numpy array,
            and a length 2N real numpy array, where N is the number of modes
            the operator acts on.
    """
    if not operator.is_gaussian():
        raise ValueError("Hamiltonian must be Gaussian "
                         "(quadratic in the quadrature operators).")

    if not is_hermitian(operator):
        raise ValueError("Hamiltonian must be Hermitian.")

    num_modes = max([op[0] for term in operator.terms for op in term])+1
    A = np.zeros([2*num_modes, 2*num_modes])
    d = np.zeros([2*num_modes])
    for term, coeff in operator.terms.items():
        c = coeff.real
        if len(term) == 2:
            if term[0][1] == term[1][1]:
                if term[0][1] == 'q':
                    A[term[0][0], term[1][0]] = c
                elif term[0][1] == 'p':
                    A[num_modes+term[0][0], num_modes+term[1][0]] = c
            else:
                if term[0][1] == 'q':
                    A[term[0][0], num_modes+term[1][0]] = c
                elif term[0][1] == 'p':
                    A[num_modes+term[0][0], term[1][0]] = c
        elif len(term) == 1:
            if term[0][1] == 'q':
                d[num_modes+term[0][0]] = -c
            elif term[0][1] == 'p':
                d[term[0][0]] = c

    A += A.T
    return A, d


class GaussianPropagation(Decomposition):
    r"""Propagate the specified qumodes by a bosonic Gaussian Hamiltonians.

    A Gaussian Hamiltonian is any combination of quadratic operators
    that can be written in quadratic form:

    .. math:: H = \frac{1}{2}\mathbf{r}A\mathbf{r} + \mathbf{r}^T \mathbf{d}

    where:

    * :math:`A\in\mathbb{R}^{2N\times 2N}` is a symmetric matrix,
    * :math:`\mathbf{d}\in\mathbb{R}^{2N}` is a real vector, and
    * :math:`\mathbf{r} = (\x_1,\dots,\x_N,\p_1,\dots,\p_N)` is the vector
      of means in :math:`xp`-ordering.

    This operation calculates the real symmetric matrix of quadratic coefficients
    of the quadrature operators, and calculates the corresponding Gaussian symplectic
    transformation via the following relation:

    .. math:: S = e^{\Omega A t \hbar}

    where

    * :math:`\Omega=\begin{bmatrix}0&I_N\\-I_N&0\end{bmatrix}\in\mathbb{R}^{2N\times 2N}`
      is the symplectic matrix,

    * :math:`\hbar` is the convention chosen in the definition of the quadrature
      operators, :math:`[\x,\p]=i\hbar`.

    Depending on whether the resulting symplectic transformation is passive
    (photon-preserving) or active (non-photon preserving), the Clements or
    Bloch-Messiah decomposition in Strawberry Fields is then used to decompose
    the Hamiltonian into a set of CV gates.

    Args:
        operator (BosonOperator, QuadOperator): a bosonic Gaussian Hamiltonian
        t (float): the time propagation value. If not provided, default value is 1.
        mode (str): By default, ``mode='local'`` and the Hamiltonian is assumed to apply to only
            the applied qumodes (q[i], q[j],...). I.e., a_0 applies to q[i], a_1 applies to q[j].
            If instead ``mode='global'``, the Hamiltonian is instead applied to the entire register;
            i.e., a_0 applied to q[0], applies to q[1], etc.
        hbar (float): the value of :math:`\hbar` used in the definition of the :math:`\x`
            and :math:`\p` quadrature operators. Note that if used inside of an engine
            context, the hbar value of the engine will override this keyword argument.
    """
    ns = None
    def __init__(self, operator, t=1, mode='local', hbar=None):
        super().__init__([t, operator])

        try:
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

        self.S = expm(sympmat(self.ns) @ A * t * self.hbar)

        self.disp = False
        if not np.all(d == 0.):
            self.disp = True
            if np.all(A == 0.):
                self.d = d*self.hbar*t
            else:
                if np.linalg.cond(A) < 1/sys.float_info.epsilon:
                    s = inv(A) @ d
                    tmp = s - self.S.T @ s
                    self.d = np.zeros([2*self.ns])
                    self.d[self.ns:] = tmp[:self.ns]
                    self.d[:self.ns] = tmp[self.ns:]
                else:
                    self.disp = False

    def decompose(self, reg):
        cmds = []
        cmds += [Command(GaussianTransform(self.S, hbar=self.hbar), reg, decomp=True)]
        if self.disp:
            cmds += [Command(Xgate(x), reg, decomp=True) for x in self.d[:self.ns] if x != 0.]
            cmds += [Command(Zgate(z), reg, decomp=True) for z in self.d[self.ns:] if z != 0.]
        return cmds
