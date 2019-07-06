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
import pytest

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


class TestSingularCoefficients:
    """Tests using singular Hamiltonians"""
    t = 0.432

    def test_singular_coefficients(self):
        """Test that H=p^2/2+q has displacement (q,t)=(-t^2,-t)"""
        prog = sf.Program(1)
        eng = sf.Engine("gaussian")

        H = QuadOperator('p0 p0', 0.5) + QuadOperator('q0')

        with prog.context as q:
            GaussianPropagation(H, self.t) | q[0]

        state = eng.run(prog).state
        res = state.means()
        expected = [-self.t**2/2, -self.t]
        assert np.allclose(res, expected)


class TestSingleModeGaussianGates:
    """Tests using single mode Gaussian gates"""

    def test_squeezing(self, hbar):
        """Test squeezing gives correct means and cov"""
        prog = sf.Program(1)
        eng = sf.Engine("gaussian")

        x = 0.2
        p = 0.3
        r = 0.42
        phi = 0.123
        H, t = squeezing(r, phi, hbar=hbar)

        with prog.context as q:
            Xgate(x) | q[0]
            Zgate(p) | q[0]
            GaussianPropagation(H, t) | q[0]

        state = eng.run(prog).state

        # test the covariance matrix
        res = state.cov()
        S = rot(phi/2) @ np.diag(np.exp([-r, r])) @ rot(phi/2).T
        V = S @ S.T * hbar/2
        assert np.allclose(res, V)

        # test the vector of means
        res = state.means()
        exp = S @ np.array([x, p])
        assert np.allclose(res, exp)

    def test_rotation(self, hbar):
        """Test rotation gives correct means and cov"""
        prog = sf.Program(1)
        eng = sf.Engine("gaussian")

        x = 0.2
        p = 0.3
        phi = 0.123
        H, t = rotation(phi, hbar=hbar)

        with prog.context as q:
            Xgate(x) | q[0]
            Zgate(p) | q[0]
            Sgate(2) | q[0]
            GaussianPropagation(H, t) | q[0]

        state = eng.run(prog).state

        # test the covariance matrix
        res = state.cov()
        V = np.diag([np.exp(-4), np.exp(4)])*hbar/2
        expected = rot(phi) @ V @ rot(phi).T
        assert np.allclose(res, expected)

        # test the vector of means
        res = state.means()
        exp = rot(phi) @ np.diag(np.exp([-2, 2])) @ np.array([x, p])
        assert np.allclose(res, exp)

    def test_quadratic_phase(self, hbar):
        """Test quadratic phase gives correct means and cov"""
        prog = sf.Program(1)
        eng = sf.Engine("gaussian")

        x = 0.2
        p = 0.3
        s = 0.432
        H, t = quadratic_phase(s)

        with prog.context as q:
            Xgate(x) | q[0]
            Zgate(p) | q[0]
            GaussianPropagation(H, t) | q[0]

        state = eng.run(prog).state

        # test the covariance matrix
        res = state.cov()
        expected = np.array([[1, s], [s, 1+s**2]])*hbar/2
        assert np.allclose(res, expected)

        # test the vector of means
        res = state.means()
        expected = np.array([x, p+s*x])
        assert np.allclose(res, expected)

    def test_displacement(self, hbar):
        """Test displacement gives correct means and cov"""
        prog = sf.Program(1)
        eng = sf.Engine("gaussian")

        a = 0.2+0.3j
        H, t = displacement(a, hbar=hbar)

        with prog.context as q:
            GaussianPropagation(H, t) | q[0]

        state = eng.run(prog).state

        # test the covariance matrix
        res = state.cov()
        expected = np.identity(2)*hbar/2
        assert np.allclose(res, expected)

        # test the vector of means
        res = state.means()
        expected = np.array([a.real, a.imag])*np.sqrt(2*hbar)
        assert np.allclose(res, expected)


def init_layer(q):
    """Create an initial state with defined
    squeezing and displacement"""
    Xgate(0.2) | q[0]
    Xgate(0.4) | q[1]
    Zgate(0.3) | q[1]
    Sgate(0.1) | q[0]
    Sgate(-0.1) | q[1]
    return q


class TestTwoModeGaussianGatesLocal:
    """Tests for two mode Gaussian gates in local mode"""
    r = 0.2
    th = 0.42
    phi = 0.123

    def H_circuit(self, H, t):
        """Test circuit for Gaussian Hamiltonian"""
        prog = sf.Program(2)
        eng = sf.Engine("gaussian")

        with prog.context as q:
            q = init_layer(q)
            GaussianPropagation(H, t) | q

        state = eng.run(prog).state
        return state.means(), state.cov()

    def ref_circuit(self, gate):
        """Reference circuit for Gaussian gate"""
        prog = sf.Program(2)
        eng = sf.Engine("gaussian")

        with prog.context as q:
            # pylint: disable=pointless-statement
            q = init_layer(q)
            gate | q

        state = eng.run(prog).state
        return state.means(), state.cov()

    def test_beamsplitter(self, hbar):
        """Test beamsplitter produces correct cov and means"""

        H, t = beamsplitter(self.th, self.phi, hbar=hbar)
        resD, resV = self.H_circuit(H, t)

        gate = BSgate(self.th, self.phi)
        expD, expV = self.ref_circuit(gate)

        # test the covariance matrix
        assert np.allclose(resV, expV)
        # test the vector of means
        assert np.allclose(resD, expD)

    def test_two_mode_squeezing(self, hbar):
        """Test S2gate produces correct cov and means"""
        H, t = two_mode_squeezing(self.r, self.phi, hbar=hbar)
        resD, resV = self.H_circuit(H, t)

        gate = S2gate(self.r, self.phi)
        expD, expV = self.ref_circuit(gate)

        # test the covariance matrix
        assert np.allclose(resV, expV)
        # test the vector of means
        assert np.allclose(resD, expD)

    def test_controlled_addition(self):
        """Test CXgate produces correct cov and means"""
        H, t = controlled_addition(self.r)
        resD, resV = self.H_circuit(H, t)

        gate = CXgate(self.r)
        expD, expV = self.ref_circuit(gate)

        # test the covariance matrix
        assert np.allclose(resV, expV)
        # test the vector of means
        assert np.allclose(resD, expD)

    def test_controlled_phase(self):
        """Test CZgate produces correct cov and means"""
        H, t = controlled_phase(self.r)
        resD, resV = self.H_circuit(H, t)

        gate = CZgate(self.r)
        expD, expV = self.ref_circuit(gate)

        # test the covariance matrix
        assert np.allclose(resV, expV)
        # test the vector of means
        assert np.allclose(resD, expD)


class TestTwoModeGaussianGatesGlobal:
    """Tests for two mode Gaussian gates in global mode"""
    r = 0.2
    th = 0.42
    phi = 0.123

    def H_circuit(self, H, t):
        """Test circuit for Gaussian Hamiltonian"""
        prog = sf.Program(3)
        eng = sf.Engine("gaussian")

        with prog.context as q:
            # pylint: disable=pointless-statement
            q = init_layer(q)
            Xgate(0.1) | q[2]
            Sgate(0.1) | q[2]
            GaussianPropagation(H, t, mode='global') | q

        state = eng.run(prog).state
        return state.means(), state.cov()

    def ref_circuit(self, gate, qm):
        """Reference circuit for Gaussian gate"""
        prog = sf.Program(3)
        eng = sf.Engine("gaussian")

        with prog.context as q:
            # pylint: disable=pointless-statement
            q = init_layer(q)
            Xgate(0.1) | q[2]
            Sgate(0.1) | q[2]
            gate | qm

        state = eng.run(prog).state
        return state.means(), state.cov()

    def test_single_mode_gate(self, hbar):
        """Test Rgate gives correct means and cov in global mode"""
        H, t = rotation(self.phi, mode=1, hbar=hbar)
        resD, resV = self.H_circuit(H, t)

        gate = Rgate(self.phi)
        expD, expV = self.ref_circuit(gate, 1)

        # test the covariance matrix
        assert np.allclose(resV, expV)
        # test the vector of means
        assert np.allclose(resD, expD)

    def test_two_mode_gate(self, hbar):
        """Test S2gate gives correct means and cov in global mode"""
        H, t = two_mode_squeezing(self.r, self.phi, mode1=0, mode2=2, hbar=hbar)
        resD, resV = self.H_circuit(H, t)

        gate = S2gate(self.r, self.phi)
        expD, expV = self.ref_circuit(gate, [0, 2])

        # test the covariance matrix
        assert np.allclose(resV, expV)
        # test the vector of means
        assert np.allclose(resD, expD)


class TestQuadraticAndLinear:
    """Tests for Hamiltonians with quadratic and linear coefficients"""
    x0 = 1
    p0 = 0.5
    F = 2
    t = 3
    dt = 0.02

    def displaced_oscillator_soln(self, t):
        """The solution to the forced quantum oscillator"""
        st = np.sin(t)
        ct = np.cos(t)
        x = self.p0*st + (self.x0-self.F)*ct + self.F
        p = (self.F-self.x0)*st + self.p0*ct
        return np.array([x, p])

    def test_displaced_oscillator(self):
        """Test that a forced quantum oscillator produces the correct
        self.logTestName()
        trajectory in the phase space"""
        H = QuadOperator('q0 q0', 0.5)
        H += QuadOperator('p0 p0', 0.5)
        H -= QuadOperator('q0', self.F)

        res = []
        tlist = np.arange(0, self.t, self.dt)

        eng = sf.Engine("gaussian")

        for idx, t in enumerate(tlist): #pylint: disable=unused-variable
            prog = sf.Program(1)

            with prog.context as q:
                Xgate(self.x0) | q[0]
                Zgate(self.p0) | q[0]
                GaussianPropagation(H, self.t) | q

            state = eng.run(prog).state
            eng.reset()
            res.append(state.means().tolist())

        res = np.array(res)[-1]
        expected = self.displaced_oscillator_soln(self.t)
        assert np.allclose(res, expected)


class TestToolchain:
    """Tests for the GaussianPropagation toolchain"""
    phi = 0.123

    def test_non_hermitian(self):
        """test exception if H is not hermitian"""
        with pytest.raises(ValueError):
            H = QuadOperator('q0', 1+2j)
            GaussianPropagation(H)
