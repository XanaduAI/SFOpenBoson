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


class TestDisplacement(unittest.TestCase):
    def setUp(self):
        self.alpha = 0.452+0.12j
        self.hbar = 2

    def test_identity(self):
        H, r = displacement(0)
        self.assertEqual(H, BosonOperator.identity())
        self.assertEqual(r, 0)

    def test_hermitian(self):
        H, r = displacement(self.alpha)
        self.assertTrue(is_hermitian(H))
        self.assertTrue(is_hermitian(get_quad_operator(H)))

    def test_gaussian(self):
        H, r = displacement(self.alpha)
        res = get_quad_operator(H).is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        H, r = displacement(self.alpha)
        self.assertEqual(r, np.abs(self.alpha))

    def test_coefficients(self):
        H, r = displacement(self.alpha)
        phi = np.angle(self.alpha)

        for term, coeff in H.terms.items():
            self.assertEqual(len(term), 1)
            j = 1-term[0][1]
            expected = (-1)**j * np.exp(1j*phi*(-1)**j)
            self.assertEqual(coeff, 1j*expected)


class TestXDisplacement(unittest.TestCase):
    def setUp(self):
        self.x = 0.452
        self.hbar = 2

    def test_identity(self):
        H, r = xdisplacement(0)
        self.assertEqual(r, 0)

    def test_hermitian(self):
        H, r = xdisplacement(self.x, hbar=self.hbar)
        self.assertTrue(is_hermitian(H))
        self.assertTrue(is_hermitian(get_boson_operator(H, hbar=self.hbar)))

    def test_gaussian(self):
        H, r = xdisplacement(self.x, hbar=self.hbar)
        res = H.is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        H, r = xdisplacement(self.x, hbar=self.hbar)
        self.assertEqual(r, self.x)

    def test_coefficients(self):
        H, r = xdisplacement(self.x, hbar=self.hbar)
        self.assertEqual(H, QuadOperator('p0', 1/self.hbar))


class TestZDisplacement(unittest.TestCase):
    def setUp(self):
        self.p = 0.452
        self.hbar = 2

    def test_identity(self):
        H, r = zdisplacement(0)
        self.assertEqual(r, 0)

    def test_hermitian(self):
        H, r = zdisplacement(self.p, hbar=self.hbar)
        self.assertTrue(is_hermitian(H))
        self.assertTrue(is_hermitian(get_boson_operator(H, hbar=self.hbar)))

    def test_gaussian(self):
        H, r = zdisplacement(self.p, hbar=self.hbar)
        res = H.is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        H, r = zdisplacement(self.p, hbar=self.hbar)
        self.assertEqual(r, self.p)

    def test_coefficients(self):
        H, r = zdisplacement(self.p, hbar=self.hbar)
        self.assertEqual(H, -QuadOperator('q0', 1/self.hbar))


class TestRotation(unittest.TestCase):
    def setUp(self):
        self.phi = 0.452
        self.hbar = 2

    def test_identity(self):
        H, r = rotation(0)
        self.assertEqual(r, 0)

    def test_hermitian(self):
        H, r = rotation(self.phi)
        self.assertTrue(is_hermitian(H))
        self.assertTrue(is_hermitian(get_quad_operator(H)))

    def test_gaussian(self):
        H, r = rotation(self.phi)
        res = get_quad_operator(H).is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        H, r = rotation(self.phi)
        self.assertEqual(r, self.phi)

    def test_coefficients(self):
        H, r = rotation(self.phi)
        self.assertEqual(H, -BosonOperator('0^ 0'))

    def test_quad_form(self):
        H, r = rotation(self.phi, mode=1)
        H = normal_ordered_quad(get_quad_operator(H, hbar=self.hbar), hbar=self.hbar)
        expected = QuadOperator('q1 q1', -1/(2*self.hbar))
        expected += QuadOperator('p1 p1', -1/(2*self.hbar))
        expected += QuadOperator('', 1/(self.hbar))
        self.assertEqual(H, expected)


class TestSqueezing(unittest.TestCase):
    def setUp(self):
        self.hbar = 2
        self.r = 0.242
        self.phi = 0.452
        self.H, self.t = squeezing(self.r, self.phi)

    def test_identity(self):
        H, t = squeezing(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_quad_operator(self.H)))

    def test_gaussian(self):
        res = get_quad_operator(self.H).is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        self.assertEqual(self.t, self.r)

    def test_quad_form(self):
        H, t = squeezing(2, mode=1)
        H = normal_ordered_quad(get_quad_operator(H, hbar=self.hbar), hbar=self.hbar)
        expected = QuadOperator('q1 p1', -1/self.hbar)
        expected += QuadOperator('', 0.5j)
        self.assertEqual(H, expected)


class TestQuadraticPhase(unittest.TestCase):
    def setUp(self):
        self.hbar = 2
        self.s = 0.242
        self.H, self.t = quadratic_phase(self.s, hbar=self.hbar)

    def test_identity(self):
        H, t = quadratic_phase(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_boson_operator(self.H, hbar=self.hbar)))

    def test_gaussian(self):
        res = self.H.is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        self.assertEqual(self.t, self.s)

    def test_boson_form(self):
        H = normal_ordered_boson(get_boson_operator(self.H, hbar=self.hbar))
        expected = BosonOperator('0 0', -0.25)
        expected += BosonOperator('0^ 0', -0.5)
        expected += BosonOperator('0^ 0^', -0.25)
        expected += BosonOperator('', -0.25)
        self.assertEqual(H, expected)


class TestBeamsplitter(unittest.TestCase):
    def setUp(self):
        self.hbar = 2
        self.theta = 0.242
        self.phi = 0.452
        self.H, self.t = beamsplitter(self.theta, self.phi)

    def test_identity(self):
        H, t = beamsplitter(0, 0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_quad_operator(self.H)))

    def test_gaussian(self):
        res = get_quad_operator(self.H).is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        self.assertEqual(self.t, self.theta)

    def test_quad_form(self):
        H, t = beamsplitter(np.pi/4, np.pi/2, mode1=1, mode2=3)
        H = normal_ordered_quad(get_quad_operator(H, hbar=self.hbar), hbar=self.hbar)
        expected = QuadOperator('q1 q3', -1/self.hbar)
        expected += QuadOperator('p1 p3', -1/self.hbar)
        self.assertEqual(H, expected)


class TestTwoModeSqueezing(unittest.TestCase):
    def setUp(self):
        self.hbar = 2
        self.r = 0.242
        self.phi = 0.452
        self.H, self.t = two_mode_squeezing(self.r, self.phi)

    def test_identity(self):
        H, t = two_mode_squeezing(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_quad_operator(self.H)))

    def test_gaussian(self):
        res = get_quad_operator(self.H).is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        self.assertEqual(self.t, self.r)

    def test_quad_form(self):
        H, t = two_mode_squeezing(2, mode1=1, mode2=3)
        H = normal_ordered_quad(get_quad_operator(H, hbar=self.hbar), hbar=self.hbar)
        expected = QuadOperator('q1 p3', 1/self.hbar)
        expected += QuadOperator('p1 q3', 1/self.hbar)
        self.assertEqual(H, expected)


class TestControlledAddition(unittest.TestCase):
    def setUp(self):
        self.hbar = 2
        self.s = 0.242
        self.H, self.t = controlled_addition(self.s, hbar=self.hbar)

    def test_identity(self):
        H, t = controlled_addition(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_boson_operator(self.H, hbar=self.hbar)))

    def test_gaussian(self):
        res = self.H.is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        self.assertEqual(self.t, self.s)

    def test_boson_form(self):
        H = normal_ordered_boson(get_boson_operator(self.H, hbar=self.hbar))
        expected = BosonOperator('0 1', -0.5j)
        expected += BosonOperator('0 1^', 0.5j)
        expected += BosonOperator('0^ 1', -0.5j)
        expected += BosonOperator('0^ 1^', 0.5j)
        self.assertEqual(H, expected)


class TestControlledPhase(unittest.TestCase):
    def setUp(self):
        self.hbar = 2
        self.s = 0.242
        self.H, self.t = controlled_phase(self.s, hbar=self.hbar)

    def test_identity(self):
        H, t = controlled_phase(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_boson_operator(self.H, hbar=self.hbar)))

    def test_gaussian(self):
        res = self.H.is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        self.assertEqual(self.t, self.s)

    def test_boson_form(self):
        H = normal_ordered_boson(get_boson_operator(self.H, hbar=self.hbar))
        expected = BosonOperator('0 1', -0.5)
        expected += BosonOperator('0 1^', -0.5)
        expected += BosonOperator('0^ 1', -0.5)
        expected += BosonOperator('0^ 1^', -0.5)
        self.assertEqual(H, expected)


class TestCubicPhase(unittest.TestCase):
    def setUp(self):
        self.hbar = 2
        self.gamma = 0.242
        self.H, self.t = cubic_phase(self.gamma, hbar=self.hbar)

    def test_identity(self):
        H, t = cubic_phase(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_boson_operator(self.H, hbar=self.hbar)))

    def test_gaussian(self):
        res = self.H.is_gaussian()
        self.assertFalse(res)

    def test_time(self):
        self.assertEqual(self.t, self.gamma)


class TestKerr(unittest.TestCase):
    def setUp(self):
        self.hbar = 2
        self.kappa = 0.242
        self.H, self.t = kerr(self.kappa)

    def test_identity(self):
        H, t = two_mode_squeezing(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_quad_operator(self.H)))

    def test_gaussian(self):
        res = get_quad_operator(self.H).is_gaussian()
        self.assertFalse(res)

    def test_time(self):
        self.assertEqual(self.t, self.kappa)


if __name__ == '__main__':
    unittest.main()
