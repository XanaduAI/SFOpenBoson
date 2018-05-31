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

from SFopenfermion.hamiltonians import *
from SFopenfermion.ops import *


class TestQuadraticCoefficients(unittest.TestCase):
    def setUp(self):
        self.hbar = 2

    def test_non_hermitian(self):
        with self.assertRaisesRegex(ValueError, "Hamiltonian must be Hermitian"):
            A, d = quadratic_coefficients(QuadOperator('q0 p0'))

    def test_non_gaussian(self):
        with self.assertRaisesRegex(ValueError, "Hamiltonian must be Gaussian"):
            A, d = quadratic_coefficients(QuadOperator('q0 p0 p1'))

    def test_displacement_vector(self):
        H = QuadOperator('q0', -0.432) + QuadOperator('p0', 3.213)
        A, d = quadratic_coefficients(H)
        expected_A = np.zeros([2, 2])
        expected_d = np.array([3.213, 0.432])
        self.assertTrue(np.allclose(A, expected_A))
        self.assertTrue(np.allclose(d, expected_d))

        A, d = quadratic_coefficients(QuadOperator('q0 q0'))
        expected = np.array([0, 0])
        self.assertTrue(np.allclose(d, expected))

        A, d = quadratic_coefficients(QuadOperator('p0 p1'))
        expected = np.array([0, 0, 0, 0])
        self.assertTrue(np.allclose(d, expected))

    def test_Dgate_displacement(self):
        a = 0.23-0.432j
        H, t = displacement(a, hbar=self.hbar)
        res, d = quadratic_coefficients(get_quad_operator(H, self.hbar))
        expected = np.array([a.real, a.imag])*np.sqrt(2*self.hbar)/t
        self.assertTrue(np.allclose(d, expected))
        self.assertTrue(np.allclose(a, t*(d[0]+d[1]*1j)/np.sqrt(2*self.hbar)))

    def test_Xgate_displacement(self):
        x = 0.1234
        H, t = xdisplacement(x)
        res, d = quadratic_coefficients(H)
        expected = np.array([1, 0])
        self.assertTrue(np.allclose(d, expected))

    def test_Zgate_displacement(self):
        z = 0.654
        H, t = zdisplacement(z)
        res, d = quadratic_coefficients(H)
        expected = np.array([0, 1])
        self.assertTrue(np.allclose(d, expected))

    def test_rotation_coeff(self):
        # one mode
        H, t = rotation(0.23, hbar=self.hbar)
        res, d = quadratic_coefficients(get_quad_operator(H, self.hbar))
        expected = -np.diag(np.array([1, 1]))
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([2])))

        # two modes
        H, t = rotation(0.23, mode=1, hbar=self.hbar)
        res, d = quadratic_coefficients(get_quad_operator(H, self.hbar))
        expected = np.zeros([4, 4])
        expected[1, 1] = -1
        expected[3, 3] = -1
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))

    def test_squeeze_coeff(self):
        # one mode
        H, t = squeezing(0.23, hbar=self.hbar)
        res, d = quadratic_coefficients(get_quad_operator(H, self.hbar))
        expected = -np.array([[0, 1], [1, 0]])
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([2])))

        # two modes
        H, t = squeezing(0.23, mode=1, hbar=self.hbar)
        res, d = quadratic_coefficients(get_quad_operator(H, self.hbar))
        expected = np.zeros([4, 4])
        expected[1, 3] = -1
        expected[3, 1] = -1
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))

    def test_quadratic_phase_coeff(self):
        # one mode
        H, t = quadratic_phase(0.23)
        res, d = quadratic_coefficients(H)
        expected = np.zeros([2, 2])
        expected[0, 0] = -1
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([2])))

        # two modes
        H, t = quadratic_phase(0.23, mode=1)
        res, d = quadratic_coefficients(H)
        expected = np.zeros([4, 4])
        expected[1, 1] = -1
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))

    def test_beamsplitter_coeff(self):
        # arbitrary beamsplitter
        theta = 0.5423
        phi = 0.3242
        H, t = beamsplitter(theta, phi, hbar=self.hbar)
        res, d = quadratic_coefficients(get_quad_operator(H, hbar=self.hbar))
        expected = np.zeros([4, 4])
        expected[0, 3] = expected[3, 0] = -np.cos(np.pi-phi)
        expected[1, 2] = expected[2, 1] = np.cos(np.pi-phi)
        expected[0, 1] = expected[1, 0] = -np.sin(np.pi-phi)
        expected[2, 3] = expected[3, 2] = -np.sin(np.pi-phi)
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))

    def test_two_mode_squeeze_coeff(self):
        H, t = two_mode_squeezing(0.23, hbar=self.hbar)
        res, d = quadratic_coefficients(get_quad_operator(H, self.hbar))
        expected = np.fliplr(np.diag([1]*4))
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))

    def test_controlled_addition(self):
        H, t = controlled_addition(0.23)
        res, d = quadratic_coefficients(H)
        expected = np.fliplr(np.diag([1, 0, 0, 1]))
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))

    def test_controlled_phase(self):
        H, t = controlled_phase(0.23)
        res, d = quadratic_coefficients(H)
        expected = np.zeros([4, 4])
        expected[0, 1] = expected[1, 0] = -1
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))


if __name__ == '__main__':
    unittest.main()
