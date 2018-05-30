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

import unittest

import numpy as np
from scipy.linalg import expm

from openfermion.ops import *
from openfermion.utils import is_hermitian
from openfermion.transforms import *

import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import squeezed_state
from strawberryfields.backends.shared_ops import rotation_matrix as rot

from SFopenfermion.hamiltonians import *
from SFopenfermion.ops import *


class TestSingleModeGaussianGates(unittest.TestCase):
    def setUp(self):
        self.hbar = 2
        self.eng, _ = sf.Engine(1, hbar=self.hbar)

    def test_squeezing(self):
        self.eng.reset()
        q = self.eng.register

        x = 0.2
        p = 0.3
        r = 0.42
        phi = 0.123
        H, t = squeezing(r, phi)

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
        self.eng.reset()
        q = self.eng.register

        x = 0.2
        p = 0.3
        phi = 0.123
        H, t = rotation(phi)

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
        self.eng.reset()
        q = self.eng.register

        x = 0.2
        p = 0.3
        s = 0.432
        H, t = quadratic_phase(s, hbar=self.hbar)

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
        self.eng.reset()
        q = self.eng.register

        a = 0.2+0.3j
        H, t = displacement(a)

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
    Xgate(0.2) | q[0]
    Xgate(0.4) | q[1]
    Zgate(0.3) | q[1]
    Sgate(0.1) | q[0]
    Sgate(-0.1) | q[1]
    return q


class TestTwoModeGaussianGatesLocal(unittest.TestCase):
    def setUp(self):
        self.hbar = 2.
        self.eng, _ = sf.Engine(2, hbar=self.hbar)

        self.r = 0.2
        self.th = 0.42
        self.phi = 0.123

    def H_circuit(self, H, t):
        self.eng.reset()
        q = self.eng.register
        with self.eng:
            q = init_layer(q)
            GaussianPropagation(H, t) | q

        state = self.eng.run('gaussian')
        return state.means(), state.cov()

    def ref_circuit(self, gate):
        self.eng.reset()
        q = self.eng.register
        with self.eng:
            q = init_layer(q)
            gate | q

        state = self.eng.run('gaussian')
        return state.means(), state.cov()

    def test_beamsplitter(self):
        self.eng.reset()
        q = self.eng.register

        H, t = beamsplitter(self.th, self.phi)
        resD, resV = self.H_circuit(H, t)

        gate = BSgate(self.th, self.phi)
        expD, expV = self.ref_circuit(gate)

        # test the covariance matrix
        self.assertTrue(np.allclose(resV, expV))
        # test the vector of means
        self.assertTrue(np.allclose(resD, expD))

    def test_two_mode_squeezing(self):
        self.eng.reset()
        q = self.eng.register

        H, t = two_mode_squeezing(self.r, self.phi)
        resD, resV = self.H_circuit(H, t)

        gate = S2gate(self.r, self.phi)
        expD, expV = self.ref_circuit(gate)

        # test the covariance matrix
        self.assertTrue(np.allclose(resV, expV))
        # test the vector of means
        self.assertTrue(np.allclose(resD, expD))

    def test_controlled_addition(self):
        self.eng.reset()
        q = self.eng.register

        H, t = controlled_addition(self.r)
        resD, resV = self.H_circuit(H, t)

        gate = CXgate(self.r)
        expD, expV = self.ref_circuit(gate)

        # test the covariance matrix
        self.assertTrue(np.allclose(resV, expV))
        # test the vector of means
        self.assertTrue(np.allclose(resD, expD))

    def test_controlled_phase(self):
        self.eng.reset()
        q = self.eng.register

        H, t = controlled_phase(self.r)
        resD, resV = self.H_circuit(H, t)

        gate = CZgate(self.r)
        expD, expV = self.ref_circuit(gate)

        # test the covariance matrix
        self.assertTrue(np.allclose(resV, expV))
        # test the vector of means
        self.assertTrue(np.allclose(resD, expD))


class TestTwoModeGaussianGatesGlobal(unittest.TestCase):
    def setUp(self):
        self.hbar = 2.
        self.eng, _ = sf.Engine(3, hbar=self.hbar)
        self.r = 0.2
        self.th = 0.42
        self.phi = 0.123

    def H_circuit(self, H, t):
        self.eng.reset()
        q = self.eng.register
        with self.eng:
            q = init_layer(q)
            Xgate(0.1) | q[2]
            Sgate(0.1) | q[2]
            GaussianPropagation(H, t, mode='global') | q

        state = self.eng.run('gaussian')
        return state.means(), state.cov()

    def ref_circuit(self, gate, qm):
        self.eng.reset()
        q = self.eng.register
        with self.eng:
            q = init_layer(q)
            Xgate(0.1) | q[2]
            Sgate(0.1) | q[2]
            gate | qm

        state = self.eng.run('gaussian')
        return state.means(), state.cov()

    def test_single_mode_gate(self):
        self.eng.reset()
        q = self.eng.register

        H, t = rotation(self.phi, mode=1)
        resD, resV = self.H_circuit(H, t)

        gate = Rgate(self.phi)
        expD, expV = self.ref_circuit(gate, q[1])

        # test the covariance matrix
        self.assertTrue(np.allclose(resV, expV))
        # test the vector of means
        self.assertTrue(np.allclose(resD, expD))

    def test_two_mode_gate(self):
        self.eng.reset()
        q = self.eng.register

        H, t = two_mode_squeezing(self.r, self.phi, mode1=0, mode2=2)
        resD, resV = self.H_circuit(H, t)

        gate = S2gate(self.r, self.phi)
        expD, expV = self.ref_circuit(gate, (q[0], q[2]))

        # test the covariance matrix
        self.assertTrue(np.allclose(resV, expV))
        # test the vector of means
        self.assertTrue(np.allclose(resD, expD))


class TestQuadraticAndLinear(unittest.TestCase):
    def setUp(self):
        self.hbar = 2.
        self.eng, _ = sf.Engine(1, hbar=self.hbar)
        self.x0 = 1
        self.p0 = 0.5
        self.F = 2
        self.t = 3
        self.dt = 0.02

    def displaced_oscillator_soln(self, t):
        st = np.sin(t*self.hbar)
        ct = np.cos(t*self.hbar)
        x = self.p0*st + (self.x0-self.F)*ct + self.F
        p = (self.F-self.x0)*st + self.p0*ct
        return np.array([x, p])

    def test_displaced_oscillator(self):
        H = QuadOperator('q0 q0', 0.5)
        H += QuadOperator('p0 p0', 0.5)
        H -= QuadOperator('q0', self.F)

        res = []
        tlist = np.arange(0, self.t, self.dt)

        for t in tlist:
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
    def setUp(self):
        self.hbar = 2.
        self.eng, _ = sf.Engine(2, hbar=self.hbar)
        self.phi = 0.123

    def ref_circuit(self, gate):
        self.eng.reset()
        q = self.eng.register
        with self.eng:
            q = init_layer(q)
            gate | q[0]

        state = self.eng.run('gaussian')
        return state.means(), state.cov()

    def test_outside_context(self):
        H, t = rotation(self.phi)
        Ugate = GaussianPropagation(H, t, hbar=self.hbar)
        resD, resV = self.ref_circuit(Ugate)
        expD, expV = self.ref_circuit(Rgate(self.phi))

        # test the covariance matrix
        self.assertTrue(np.allclose(resV, expV))
        # test the vector of means
        self.assertTrue(np.allclose(resD, expD))

    def test_outside_context_no_hbar(self):
        with self.assertRaises(ValueError):
            H, t = rotation(self.phi)
            Ugate = GaussianPropagation(H, t)

    def test_non_hermitian(self):
        with self.assertRaises(ValueError):
            H = QuadOperator('q0', 1+2j)
            Ugate = GaussianPropagation(H, hbar=2.)


if __name__ == '__main__':
    unittest.main()
