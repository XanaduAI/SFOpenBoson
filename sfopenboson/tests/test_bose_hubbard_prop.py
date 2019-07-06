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
import pytest

import numpy as np
from scipy.linalg import expm

from openfermion.ops import BosonOperator, QuadOperator
from openfermion.transforms import get_quad_operator
from openfermion.hamiltonians import bose_hubbard

import strawberryfields as sf
from strawberryfields.ops import Fock, Sgate

from sfopenboson.ops import BoseHubbardPropagation


class TestBoseHubbardPropagation:
    """Tests for the BoseHubbardPropagation operation"""
    J = -1
    U = 1.5
    V = 1/np.sqrt(2)
    t = 1.086
    k = 20

    @pytest.fixture
    def eng(self, hbar):
        """Engine to use for the tests"""
        sf.hbar = hbar
        return sf.Engine("fock", backend_options={"cutoff_dim": 7})

    def test_1x2_no_onsite(self, eng, tol):
        """Test a 1x2 lattice Bose-Hubbard model"""
        prog = sf.Program(2)
        H = bose_hubbard(1, 2, self.J, 0)

        with prog.context as q:
            Fock(2) | q[0]
            BoseHubbardPropagation(H, self.t, self.k) | q

        state = eng.run(prog).state

        Hm = -self.J*np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        init_state = np.array([1, 0, 0])
        exp = np.abs(np.dot(expm(-1j*self.t*Hm), init_state))**2

        assert np.allclose(state.fock_prob([2, 0]), exp[0], rtol=tol)
        assert np.allclose(state.fock_prob([1, 1]), exp[1], rtol=tol)
        assert np.allclose(state.fock_prob([0, 2]), exp[2], rtol=tol)

    def test_1x2(self, eng, tol):
        """Test a 1x2 lattice Bose-Hubbard model"""
        prog = sf.Program(2)
        H = bose_hubbard(1, 2, self.J, self.U)

        with prog.context as q:
            Fock(2) | q[0]
            BoseHubbardPropagation(H, self.t, self.k) | q

        state = eng.run(prog).state

        Hm = -self.J*np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) \
            + self.U*np.diag([1, 0, 1])
        init_state = np.array([1, 0, 0])
        exp = np.abs(np.dot(expm(-1j*self.t*Hm), init_state))**2

        assert np.allclose(state.fock_prob([2, 0]), exp[0], rtol=tol)
        assert np.allclose(state.fock_prob([1, 1]), exp[1], rtol=tol)
        assert np.allclose(state.fock_prob([0, 2]), exp[2], rtol=tol)

    def test_1x2_dipole(self, eng, tol):
        """Test a 1x2 lattice Bose-Hubbard model with nearest-neighbour interactions"""
        prog = sf.Program(2)
        H = bose_hubbard(1, 2, self.J, self.U, 0, self.V)

        with prog.context as q:
            Fock(2) | q[0]
            BoseHubbardPropagation(H, self.t, self.k) | q

        state = eng.run(prog).state

        Hm = -self.J*np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) \
            + self.U*np.diag([1, 0, 1]) + self.V*np.diag([0, 1, 0])
        init_state = np.array([1, 0, 0])
        exp = np.abs(np.dot(expm(-1j*self.t*Hm), init_state))**2

        assert np.allclose(state.fock_prob([2, 0]), exp[0], rtol=tol)
        assert np.allclose(state.fock_prob([1, 1]), exp[1], rtol=tol)
        assert np.allclose(state.fock_prob([0, 2]), exp[2], rtol=tol)

    def test_1x2_tf(self, hbar, tol):
        """Test a 1x2 lattice Bose-Hubbard model using TF"""
        try:
            import tensorflow as tf
        except (ImportError, ModuleNotFoundError):
            pytest.skip("TensorFlow not installed.")

        if tf.__version__[:3] != "1.3":
            pytest.skip("Incorrect TensorFlow version")

        sf.hbar = hbar
        prog = sf.Program(2)
        H = bose_hubbard(1, 2, self.J, self.U)

        with prog.context as q:
            Fock(2) | q[0]
            BoseHubbardPropagation(H, self.t, self.k) | q

        eng = sf.Engine("tf", backend_options={"cutoff_dim": 7})
        state = eng.run(prog).state

        Hm = -self.J*np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) \
            + self.U*np.diag([1, 0, 1])
        init_state = np.array([1, 0, 0])
        exp = np.abs(np.dot(expm(-1j*self.t*Hm), init_state))**2

        assert np.allclose(state.fock_prob([2, 0]), exp[0], rtol=tol)
        assert np.allclose(state.fock_prob([1, 1]), exp[1], rtol=tol)
        assert np.allclose(state.fock_prob([0, 2]), exp[2], rtol=tol)


class TestBoseHubbardGlobalLocal:
    """Tests for the BoseHubbardPropagation
    operation local and global modes"""
    J = -1
    U = 1.5
    t = 1.086
    k = 20
    H = bose_hubbard(1, 2, J, U)

    @pytest.fixture
    def eng(self, cutoff, hbar):
        """Engine to use for the tests"""
        sf.hbar = hbar
        return sf.Engine("fock", backend_options={"cutoff_dim": cutoff})

    @pytest.mark.parametrize("cutoff", [7])
    def test_local(self, eng, tol):
        """Test a 1x2 lattice Bose-Hubbard model in local mode"""
        prog = sf.Program(3)

        with prog.context as q:
            Sgate(0.1) | q[1]
            Fock(2) | q[0]
            BoseHubbardPropagation(self.H, self.t, self.k) | (q[0], q[2])

        state = eng.run(prog, modes=[0, 2]).state

        Hm = -self.J*np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) \
            + self.U*np.diag([1, 0, 1])
        init_state = np.array([1, 0, 0])
        exp = np.abs(np.dot(expm(-1j*self.t*Hm), init_state))**2

        assert np.allclose(state.fock_prob([2, 0]), exp[0], rtol=tol)
        assert np.allclose(state.fock_prob([1, 1]), exp[1], rtol=tol)
        assert np.allclose(state.fock_prob([0, 2]), exp[2], rtol=tol)

    @pytest.mark.parametrize("cutoff", [7])
    def test_global(self, eng, tol):
        """Test a 1x2 lattice Bose-Hubbard model in global mode"""
        prog = sf.Program(3)

        with prog.context as q:
            Sgate(0.1) | q[2]
            Fock(2) | q[0]
            BoseHubbardPropagation(self.H, self.t, self.k, mode='global') | q

        state = eng.run(prog, modes=[0, 1]).state

        Hm = -self.J*np.sqrt(2)*np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) \
            + self.U*np.diag([1, 0, 1])
        init_state = np.array([1, 0, 0])
        exp = np.abs(np.dot(expm(-1j*self.t*Hm), init_state))**2

        assert np.allclose(state.fock_prob([2, 0]), exp[0], rtol=tol)
        assert np.allclose(state.fock_prob([1, 1]), exp[1], rtol=tol)
        assert np.allclose(state.fock_prob([0, 2]), exp[2], rtol=tol)

    @pytest.mark.parametrize("cutoff", [3])
    def test_circulant(self, eng, tol):
        """Test a 3-cycle Bose-Hubbard model in local mode"""
        prog = sf.Program(3)

        # tunnelling terms
        H = BosonOperator('0 1^', -self.J) + BosonOperator('0^ 1', -self.J)
        H += BosonOperator('0 2^', -self.J) + BosonOperator('0^ 2', -self.J)
        H += BosonOperator('1 2^', -self.J) + BosonOperator('1^ 2', -self.J)

        # on-site interactions
        H += BosonOperator('0^ 0 0^ 0', 0.5*self.U) - BosonOperator('0^ 0', 0.5*self.U)
        H += BosonOperator('1^ 1 1^ 1', 0.5*self.U) - BosonOperator('1^ 1', 0.5*self.U)
        H += BosonOperator('2^ 2 2^ 2', 0.5*self.U) - BosonOperator('2^ 2', 0.5*self.U)

        with prog.context as q:
            Fock(2) | q[0]
            BoseHubbardPropagation(H, self.t, 50) | q

        state = eng.run(prog).state

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

        assert np.allclose(state.fock_prob([2, 0, 0]), exp[0], rtol=tol)
        assert np.allclose(state.fock_prob([1, 1, 0]), exp[1], rtol=tol)
        assert np.allclose(state.fock_prob([1, 0, 1]), exp[2], rtol=tol)
        assert np.allclose(state.fock_prob([0, 2, 0]), exp[3], rtol=tol)
        assert np.allclose(state.fock_prob([0, 1, 1]), exp[4], rtol=tol)
        assert np.allclose(state.fock_prob([0, 0, 2]), exp[5], rtol=tol)


class TestToolchain:
    """Tests for the BoseHubbardPropagation toolchain"""
    J = -1
    U = 1.5
    t = 1.086
    k = 20
    H = bose_hubbard(1, 2, J, U)

    def test_non_hermitian(self):
        """Tests exception if non-Hermitian H"""
        with pytest.raises(ValueError):
            H = QuadOperator('q0', 1+2j)
            BoseHubbardPropagation(H, self.t, self.k)

    def test_invalid_k(self):
        """Tests exception if k<=0 or k is non-integer"""
        with pytest.raises(ValueError):
            BoseHubbardPropagation(self.H, self.t, 0)

        with pytest.raises(ValueError):
            BoseHubbardPropagation(self.H, self.t, -2)

        with pytest.raises(ValueError):
            BoseHubbardPropagation(self.H, self.t, 7.12)
