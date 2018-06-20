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
"""Tests for GaussianPropagation operation"""
# pylint: disable=expression-not-assigned
import unittest

import numpy as np
from openfermion.ops import QuadOperator

import strawberryfields as sf
from strawberryfields.ops import (BSgate,
                                  CXgate,
                                  CZgate,
                                  Rgate,
                                  Sgate,
                                  S2gate,
                                  Xgate,
                                  Zgate)

from strawberryfields.backends.shared_ops import rotation_matrix as rot

from sfopenboson.hamiltonians import (displacement,
                                      rotation,
                                      squeezing,
                                      quadratic_phase,
                                      beamsplitter,
                                      two_mode_squeezing,
                                      controlled_addition,
                                      controlled_phase)

from sfopenboson.ops import GaussianPropagation


class TestSingularCoefficients(unittest.TestCase):
    """Tests using singular Hamiltonians"""
    def setUp(self):
        """parameters"""
        self.hbar = 2
        self.eng, _ = sf.Engine(1, hbar=self.hbar)
        self.t = 0.432

    def test_singular_coefficients(self):
        """Test that H=p^2/2+q has displacement (q,t)=(-t^2,-t)"""
        self.eng.reset()
        q = self.eng.register

        H = QuadOperator('p0 p0', 0.5) + QuadOperator('q0')

        with self.eng:
            GaussianPropagation(H, self.t) | q[0]

        state = self.eng.run('gaussian')
        res = state.means()
        expected = [-self.t**2/2, -self.t]
        self.assertTrue(np.allclose(res, expected))


class TestSingleModeGaussianGates(unittest.TestCase):
    """Tests using single mode Gaussian gates"""
    def setUp(self):
        """parameters"""
        self.hbar = 2
        self.eng, _ = sf.Engine(1, hbar=self.hbar)

    def test_squeezing(self):
        """Test squeezing gives correct means and cov"""
        self.eng.reset()
        q = self.eng.register

        x = 0.2
        p = 0.3
        r = 0.42
        phi = 0.123
        H, t = squeezing(r, phi, hbar=self.hbar)

        with self.eng:
            Xgate(x) | q[0]
            Zgate(p) | q[0]
            GaussianPropagation(H, t) | q[0]

        state = self.eng.run('gaussian')

        # test the covariance matrix
        res = state.cov()
        S = rot(phi/2) @ np.diag(np.exp([-r, r])) @ rot(phi/2).T
        V = S @ S.T * self.hbar/2
        self.assertTrue(np.allclose(res, V))

        # test the vector of means
        res = state.means()
        exp = S @ np.array([x, p])
        self.assertTrue(np.allclose(res, exp))

    def test_rotation(self):
        """Test rotation gives correct means and cov"""
        self.eng.reset()
        q = self.eng.register

        x = 0.2
        p = 0.3
        phi = 0.123
        H, t = rotation(phi, hbar=self.hbar)

        with self.eng:
            Xgate(x) | q[0]
            Zgate(p) | q[0]
            Sgate(2) | q[0]
            GaussianPropagation(H, t) | q[0]

        state = self.eng.run('gaussian')

        # test the covariance matrix
        res = state.cov()
        V = np.diag([np.exp(-4), np.exp(4)])*self.hbar/2
        expected = rot(phi) @ V @ rot(phi).T
        self.assertTrue(np.allclose(res, expected))

        # test the vector of means
        res = state.means()
        exp = rot(phi) @ np.diag(np.exp([-2, 2])) @ np.array([x, p])
        self.assertTrue(np.allclose(res, exp))

    def test_quadratic_phase(self):
        """Test quadratic phase gives correct means and cov"""
        self.eng.reset()
        q = self.eng.register

        x = 0.2
        p = 0.3
        s = 0.432
        H, t = quadratic_phase(s)

        with self.eng:
            Xgate(x) | q[0]
            Zgate(p) | q[0]
            GaussianPropagation(H, t) | q[0]

        state = self.eng.run('gaussian')

        # test the covariance matrix
        res = state.cov()
        expected = np.array([[1, s], [s, 1+s**2]])*self.hbar/2
        self.assertTrue(np.allclose(res, expected))

        # test the vector of means
        res = state.means()
        expected = np.array([x, p+s*x])
        self.assertTrue(np.allclose(res, expected))

    def test_displacement(self):
        """Test displacement gives correct means and cov"""
        self.eng.reset()
        q = self.eng.register

        a = 0.2+0.3j
        H, t = displacement(a, hbar=self.hbar)

        with self.eng:
            GaussianPropagation(H, t) | q[0]

        state = self.eng.run('gaussian')

        # test the covariance matrix
        res = state.cov()
        expected = np.identity(2)*self.hbar/2
        self.assertTrue(np.allclose(res, expected))

        # test the vector of means
        res = state.means()
        expected = np.array([a.real, a.imag])*np.sqrt(2*self.hbar)
        self.assertTrue(np.allclose(res, expected))


def init_layer(q):
    """Create an initial state with defined
    squeezing and displacement"""
    Xgate(0.2) | q[0]
    Xgate(0.4) | q[1]
    Zgate(0.3) | q[1]
    Sgate(0.1) | q[0]
    Sgate(-0.1) | q[1]
    return q


class TestTwoModeGaussianGatesLocal(unittest.TestCase):
    """Tests for two mode Gaussian gates in local mode"""
    def setUp(self):
        """parameters"""
        self.hbar = 2.
        self.eng, _ = sf.Engine(2, hbar=self.hbar)

        self.r = 0.2
        self.th = 0.42
        self.phi = 0.123

    def H_circuit(self, H, t):
        """Test circuit for Gaussian Hamiltonian"""
        self.eng.reset()
        q = self.eng.register
        with self.eng:
            q = init_layer(q)
            GaussianPropagation(H, t) | q

        state = self.eng.run('gaussian')
        return state.means(), state.cov()

    def ref_circuit(self, gate):
        """Reference circuit for Gaussian gate"""
        self.eng.reset()
        q = self.eng.register
        with self.eng:
            # pylint: disable=pointless-statement
            q = init_layer(q)
            gate | q

        state = self.eng.run('gaussian')
        return state.means(), state.cov()

    def test_beamsplitter(self):
        """Test beamsplitter produces correct cov and means"""
        self.eng.reset()

        H, t = beamsplitter(self.th, self.phi, hbar=self.hbar)
        resD, resV = self.H_circuit(H, t)

        gate = BSgate(self.th, self.phi)
        expD, expV = self.ref_circuit(gate)

        # test the covariance matrix
        self.assertTrue(np.allclose(resV, expV))
        # test the vector of means
        self.assertTrue(np.allclose(resD, expD))

    def test_two_mode_squeezing(self):
        """Test S2gate produces correct cov and means"""
        # NOTE: There is currently a bug in strawberry fields,
        # where the Bloch-Messiah decomposition returns an
        # incorrect result for matrices with degenerate eigenvalues.
        # self.eng.reset()

        # H, t = two_mode_squeezing(self.r, self.phi, hbar=self.hbar)
        # resD, resV = self.H_circuit(H, t)

        # gate = S2gate(self.r, self.phi)
        # expD, expV = self.ref_circuit(gate)

        # # test the covariance matrix
        # self.assertTrue(np.allclose(resV, expV))
        # # test the vector of means
        # self.assertTrue(np.allclose(resD, expD))

    def test_controlled_addition(self):
        """Test CXgate produces correct cov and means"""
        self.eng.reset()

        H, t = controlled_addition(self.r)
        resD, resV = self.H_circuit(H, t)

        gate = CXgate(self.r)
        expD, expV = self.ref_circuit(gate)

        # test the covariance matrix
        self.assertTrue(np.allclose(resV, expV))
        # test the vector of means
        self.assertTrue(np.allclose(resD, expD))

    def test_controlled_phase(self):
        """Test CZgate produces correct cov and means"""
        self.eng.reset()

        H, t = controlled_phase(self.r)
        resD, resV = self.H_circuit(H, t)

        gate = CZgate(self.r)
        expD, expV = self.ref_circuit(gate)

        # test the covariance matrix
        self.assertTrue(np.allclose(resV, expV))
        # test the vector of means
        self.assertTrue(np.allclose(resD, expD))


class TestTwoModeGaussianGatesGlobal(unittest.TestCase):
    """Tests for two mode Gaussian gates in global mode"""
    def setUp(self):
        """parameters"""
        self.hbar = 2.
        self.eng, _ = sf.Engine(3, hbar=self.hbar)
        self.r = 0.2
        self.th = 0.42
        self.phi = 0.123

    def H_circuit(self, H, t):
        """Test circuit for Gaussian Hamiltonian"""
        self.eng.reset()
        q = self.eng.register
        with self.eng:
            # pylint: disable=pointless-statement
            q = init_layer(q)
            Xgate(0.1) | q[2]
            Sgate(0.1) | q[2]
            GaussianPropagation(H, t, mode='global') | q

        state = self.eng.run('gaussian')
        return state.means(), state.cov()

    def ref_circuit(self, gate, qm):
        """Reference circuit for Gaussian gate"""
        self.eng.reset()
        q = self.eng.register
        with self.eng:
            # pylint: disable=pointless-statement
            q = init_layer(q)
            Xgate(0.1) | q[2]
            Sgate(0.1) | q[2]
            gate | qm

        state = self.eng.run('gaussian')
        return state.means(), state.cov()

    def test_single_mode_gate(self):
        """Test Rgate gives correct means and cov in global mode"""
        self.eng.reset()
        q = self.eng.register

        H, t = rotation(self.phi, mode=1, hbar=self.hbar)
        resD, resV = self.H_circuit(H, t)

        gate = Rgate(self.phi)
        expD, expV = self.ref_circuit(gate, q[1])

        # test the covariance matrix
        self.assertTrue(np.allclose(resV, expV))
        # test the vector of means
        self.assertTrue(np.allclose(resD, expD))

    def test_two_mode_gate(self):
        """Test S2gate gives correct means and cov in global mode"""
        self.eng.reset()
        q = self.eng.register

        H, t = two_mode_squeezing(self.r, self.phi, mode1=0, mode2=2, hbar=self.hbar)
        resD, resV = self.H_circuit(H, t)

        gate = S2gate(self.r, self.phi)
        expD, expV = self.ref_circuit(gate, (q[0], q[2]))

        # test the covariance matrix
        self.assertTrue(np.allclose(resV, expV))
        # test the vector of means
        self.assertTrue(np.allclose(resD, expD))


class TestQuadraticAndLinear(unittest.TestCase):
    """Tests for Hamiltonians with quadratic and linear coefficients"""
    def setUp(self):
        """parameters"""
        self.hbar = 2.
        self.eng, _ = sf.Engine(1, hbar=self.hbar)
        self.x0 = 1
        self.p0 = 0.5
        self.F = 2
        self.t = 3
        self.dt = 0.02

    def displaced_oscillator_soln(self, t):
        """The solution to the forced quantum oscillator"""
        st = np.sin(t)
        ct = np.cos(t)
        x = self.p0*st + (self.x0-self.F)*ct + self.F
        p = (self.F-self.x0)*st + self.p0*ct
        return np.array([x, p])

    def test_displaced_oscillator(self):
        """Test that a forced quantum oscillator produces the correct
        trajectory in the phase space"""
        H = QuadOperator('q0 q0', 0.5)
        H += QuadOperator('p0 p0', 0.5)
        H -= QuadOperator('q0', self.F)

        res = []
        tlist = np.arange(0, self.t, self.dt)

        for t in tlist: #pylint: disable=unused-variable
            self.eng.reset()
            q = self.eng.register
            with self.eng:
                Xgate(self.x0) | q[0]
                Zgate(self.p0) | q[0]
                GaussianPropagation(H, self.t) | q

            state = self.eng.run('gaussian')
            res.append(state.means().tolist())

        res = np.array(res)[-1]
        expected = self.displaced_oscillator_soln(self.t)
        self.assertTrue(np.allclose(res, expected))


class TestToolchain(unittest.TestCase):
    """Tests for the GaussianPropagation toolchain"""
    def setUp(self):
        """parameters"""
        self.hbar = 2.
        self.eng, _ = sf.Engine(2, hbar=self.hbar)
        self.phi = 0.123

    def ref_circuit(self, gate):
        """Reference circuit for a unitary gate"""
        self.eng.reset()
        q = self.eng.register
        with self.eng:
            # pylint: disable=pointless-statement
            q = init_layer(q)
            gate | q[0]

        state = self.eng.run('gaussian')
        return state.means(), state.cov()

    def test_outside_context(self):
        """test setting hbar outside of engine context"""
        H, t = rotation(self.phi)
        Ugate = GaussianPropagation(H, t, hbar=self.hbar)
        resD, resV = self.ref_circuit(Ugate)
        expD, expV = self.ref_circuit(Rgate(self.phi))

        # test the covariance matrix
        self.assertTrue(np.allclose(resV, expV))
        # test the vector of means
        self.assertTrue(np.allclose(resD, expD))

    def test_outside_context_no_hbar(self):
        """test exception if no hbar outside of engine context"""
        with self.assertRaises(ValueError):
            H, t = rotation(self.phi)
            GaussianPropagation(H, t)

    def test_non_hermitian(self):
        """test exception if H is not hermitian"""
        with self.assertRaises(ValueError):
            H = QuadOperator('q0', 1+2j)
            GaussianPropagation(H, hbar=2.)


if __name__ == '__main__':
    unittest.main()
