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
"""Test Bose-Hubbard decomposition"""
# pylint: disable=expression-not-assigned, too-many-instance-attributes
import unittest

import numpy as np
from scipy.linalg import expm

from openfermion.ops import BosonOperator, QuadOperator
from openfermion.transforms import get_quad_operator
from openfermion.hamiltonians import bose_hubbard

import strawberryfields as sf
from strawberryfields.ops import Fock, Sgate

from sfopenboson.ops import BoseHubbardPropagation


class TestBoseHubbardPropagation(unittest.TestCase):
    """Tests for the BoseHubbardPropagation operation"""
    def setUp(self):
        """Parameters"""
        self.hbar = 2.
        self.eng, _ = sf.Engine(2, hbar=self.hbar)
        self.J = -1
        self.U = 1.5
        self.t = 1.086
        self.k = 20
        self.tol = 1e-2

    def test_1x2(self):
        """Test a 1x2 lattice Bose-Hubbard model"""
        self.eng.reset()
        q = self.eng.register
        H = bose_hubbard(1, 2, self.J, self.U)

        with self.eng:
            Fock(2) | q[0]
            BoseHubbardPropagation(H, self.t, self.k) | q

        state = self.eng.run('fock', cutoff_dim=7)

        Hm = -self.J*np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) \
            + self.U*np.diag([1, 0, 1])
        init_state = np.array([1, 0, 0])
        exp = np.abs(np.dot(expm(-1j*self.t*Hm), init_state))**2

        self.assertTrue(np.allclose(state.fock_prob([2, 0]), exp[0], rtol=self.tol))
        self.assertTrue(np.allclose(state.fock_prob([1, 1]), exp[1], rtol=self.tol))
        self.assertTrue(np.allclose(state.fock_prob([0, 2]), exp[2], rtol=self.tol))


class TestBoseHubbardGlobalLocal(unittest.TestCase):
    """Tests for the BoseHubbardPropagation
    operation local and global modes"""
    def setUp(self):
        self.hbar = 2.
        self.eng, _ = sf.Engine(3, hbar=self.hbar)
        self.J = -1
        self.U = 1.5
        self.t = 1.086
        self.k = 20
        self.tol = 1e-2

        self.H = bose_hubbard(1, 2, self.J, self.U)

    def test_local(self):
        """Test a 1x2 lattice Bose-Hubbard model in local mode"""
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            Sgate(0.1) | q[1]
            Fock(2) | q[0]
            BoseHubbardPropagation(self.H, self.t, self.k) | (q[0], q[2])

        state = self.eng.run('fock', cutoff_dim=7, modes=[0, 2])

        Hm = -self.J*np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) \
            + self.U*np.diag([1, 0, 1])
        init_state = np.array([1, 0, 0])
        exp = np.abs(np.dot(expm(-1j*self.t*Hm), init_state))**2

        self.assertTrue(np.allclose(state.fock_prob([2, 0]), exp[0], rtol=self.tol))
        self.assertTrue(np.allclose(state.fock_prob([1, 1]), exp[1], rtol=self.tol))
        self.assertTrue(np.allclose(state.fock_prob([0, 2]), exp[2], rtol=self.tol))

    def test_global(self):
        """Test a 1x2 lattice Bose-Hubbard model in global mode"""
        self.eng.reset()
        q = self.eng.register

        with self.eng:
            Sgate(0.1) | q[2]
            Fock(2) | q[0]
            BoseHubbardPropagation(self.H, self.t, self.k, mode='global') | q

        state = self.eng.run('fock', cutoff_dim=7, modes=[0, 1])

        Hm = -self.J*np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) \
            + self.U*np.diag([1, 0, 1])
        init_state = np.array([1, 0, 0])
        exp = np.abs(np.dot(expm(-1j*self.t*Hm), init_state))**2

        self.assertTrue(np.allclose(state.fock_prob([2, 0]), exp[0], rtol=self.tol))
        self.assertTrue(np.allclose(state.fock_prob([1, 1]), exp[1], rtol=self.tol))
        self.assertTrue(np.allclose(state.fock_prob([0, 2]), exp[2], rtol=self.tol))

    def test_circulant(self):
        """Test a 3-cycle Bose-Hubbard model in local mode"""
        self.eng.reset()
        q = self.eng.register
        # tunnelling terms
        H = BosonOperator('0 1^', -self.J) + BosonOperator('0^ 1', -self.J)
        H += BosonOperator('0 2^', -self.J) + BosonOperator('0^ 2', -self.J)
        H += BosonOperator('1 2^', -self.J) + BosonOperator('1^ 2', -self.J)

        # on-site interactions
        H += BosonOperator('0^ 0 0^ 0', 0.5*self.U) - BosonOperator('0^ 0', 0.5*self.U)
        H += BosonOperator('1^ 1 1^ 1', 0.5*self.U) - BosonOperator('1^ 1', 0.5*self.U)
        H += BosonOperator('2^ 2 2^ 2', 0.5*self.U) - BosonOperator('2^ 2', 0.5*self.U)

        with self.eng:
            Fock(2) | q[0]
            BoseHubbardPropagation(H, self.t, 50) | q

        state = self.eng.run('fock', cutoff_dim=3)

        # 2D hamiltonian
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        A2 = np.kron(A, np.identity(3)) + np.kron(np.identity(3), A)

        # change of basis matrix
        B = np.zeros([6, 9])
        B[0, 0] = B[5, 8] = B[3, 4] = 1
        B[1, 1] = B[1, 3] = 1/np.sqrt(2)
        B[2, 2] = B[2, 6] = 1/np.sqrt(2)
        B[4, 5] = B[4, 7] = 1/np.sqrt(2)

        # create Hamiltonian and add interaction term
        Hm = B @ (self.J*A2) @ B.T
        Hm[0, 0] = Hm[3, 3] = Hm[5, 5] = self.U

        init_state = np.zeros([6])
        init_state[0] = 1
        exp = np.abs(np.dot(expm(-1j*self.t*Hm), init_state))**2

        tol = 1e-1

        self.assertTrue(np.allclose(state.fock_prob([2, 0, 0]), exp[0], rtol=tol))
        self.assertTrue(np.allclose(state.fock_prob([1, 1, 0]), exp[1], rtol=tol))
        self.assertTrue(np.allclose(state.fock_prob([1, 0, 1]), exp[2], rtol=tol))
        self.assertTrue(np.allclose(state.fock_prob([0, 2, 0]), exp[3], rtol=tol))
        self.assertTrue(np.allclose(state.fock_prob([0, 1, 1]), exp[4], rtol=tol))
        self.assertTrue(np.allclose(state.fock_prob([0, 0, 2]), exp[5], rtol=tol))


class TestToolchain(unittest.TestCase):
    """Tests for the BoseHubbardPropagation toolchain"""
    def setUp(self):
        """parameters"""
        self.hbar = 2.
        self.eng, _ = sf.Engine(2, hbar=self.hbar)
        self.J = -1
        self.U = 1.5
        self.t = 1.086
        self.k = 20
        self.tol = 1e-2

        self.H = bose_hubbard(1, 2, self.J, self.U)
        self.Hquad = get_quad_operator(self.H, hbar=self.hbar)

    def test_hbar_outside_context(self):
        """Tests setting hbar outside the engine"""
        self.eng.reset()
        q = self.eng.register
        HBH = BoseHubbardPropagation(self.Hquad, self.t, self.k, hbar=self.hbar)

        with self.eng:
            # pylint: disable=pointless-statement
            Fock(2) | q[0]
            HBH | q

        state = self.eng.run('fock', cutoff_dim=7)

        Hm = -self.J*np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) \
            + self.U*np.diag([1, 0, 1])
        init_state = np.array([1, 0, 0])
        exp = np.abs(np.dot(expm(-1j*self.t*Hm), init_state))**2

        self.assertTrue(np.allclose(state.fock_prob([2, 0]), exp[0], rtol=self.tol))
        self.assertTrue(np.allclose(state.fock_prob([1, 1]), exp[1], rtol=self.tol))
        self.assertTrue(np.allclose(state.fock_prob([0, 2]), exp[2], rtol=self.tol))

    def test_outside_context_no_hbar(self):
        """Tests exception if outside the engine with no hbar"""
        with self.assertRaises(ValueError):
            BoseHubbardPropagation(self.Hquad, self.t, self.k)

    def test_non_hermitian(self):
        """Tests exception if non-Hermitian H"""
        with self.assertRaises(ValueError):
            H = QuadOperator('q0', 1+2j)
            BoseHubbardPropagation(H, self.t, self.k, hbar=self.hbar)

    def test_invalid_k(self):
        """Tests exception if k<=0 or k is non-integer"""
        with self.assertRaises(ValueError):
            BoseHubbardPropagation(self.H, self.t, 0, hbar=self.hbar)

        with self.assertRaises(ValueError):
            BoseHubbardPropagation(self.H, self.t, -2, hbar=self.hbar)

        with self.assertRaises(ValueError):
            BoseHubbardPropagation(self.H, self.t, 7.12, hbar=self.hbar)


if __name__ == '__main__':
    unittest.main()
