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
"""Tests for the CV gate Hamiltonians"""

import unittest

import numpy as np

from openfermion.ops import BosonOperator, QuadOperator
from openfermion.utils import is_hermitian, normal_ordered
from openfermion.transforms import get_boson_operator, get_quad_operator

from sfopenboson.hamiltonians import (displacement,
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


class TestDisplacement(unittest.TestCase):
    """Tests for Displacement function"""
    def setUp(self):
        """parameters"""
        self.alpha = 0.452+0.12j
        self.hbar = 2

    def test_identity(self):
        """Test alpha=0 gives identity"""
        H, r = displacement(0, hbar=self.hbar)
        self.assertEqual(H, BosonOperator.identity())
        self.assertEqual(r, 0)

    def test_hermitian(self):
        """Test output is hermitian"""
        H, _ = displacement(self.alpha, hbar=self.hbar)
        self.assertTrue(is_hermitian(H))
        self.assertTrue(is_hermitian(get_quad_operator(H)))

    def test_gaussian(self):
        """Test output is gaussian"""
        H, _ = displacement(self.alpha, hbar=self.hbar)
        res = get_quad_operator(H, hbar=self.hbar).is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        """Test time parameter is correct"""
        _, r = displacement(self.alpha)
        self.assertEqual(r, np.abs(self.alpha))

    def test_coefficients(self):
        """Test coefficients are correct"""
        H, _ = displacement(self.alpha, hbar=self.hbar)
        phi = np.angle(self.alpha)

        for term, coeff in H.terms.items():
            self.assertEqual(len(term), 1)
            j = 1-term[0][1]
            expected = (-1)**j * np.exp(1j*phi*(-1)**j)*self.hbar
            self.assertEqual(coeff, 1j*expected)


class TestXDisplacement(unittest.TestCase):
    """Tests for x displacement function"""
    def setUp(self):
        """parameters"""
        self.x = 0.452
        self.hbar = 2

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, r = xdisplacement(0)
        self.assertEqual(r, 0)

    def test_hermitian(self):
        """Test output is hermitian"""
        H, _ = xdisplacement(self.x)
        self.assertTrue(is_hermitian(H))
        self.assertTrue(is_hermitian(get_boson_operator(H, hbar=self.hbar)))

    def test_gaussian(self):
        """Test output is gaussian"""
        H, _ = xdisplacement(self.x)
        res = H.is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        """Test time parameter is correct"""
        _, r = xdisplacement(self.x)
        self.assertEqual(r, self.x)

    def test_coefficients(self):
        """Test coefficients are correct"""
        H, _ = xdisplacement(self.x)
        self.assertEqual(H, QuadOperator('p0', 1))


class TestZDisplacement(unittest.TestCase):
    """Tests for z displacement function"""
    def setUp(self):
        """Parameters"""
        self.p = 0.452
        self.hbar = 2

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, r = zdisplacement(0)
        self.assertEqual(r, 0)

    def test_hermitian(self):
        """Test output is hermitian"""
        H, _ = zdisplacement(self.p)
        self.assertTrue(is_hermitian(H))
        self.assertTrue(is_hermitian(get_boson_operator(H, hbar=self.hbar)))

    def test_gaussian(self):
        """Test output is gaussian"""
        H, _ = zdisplacement(self.p)
        res = H.is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        """Test time parameter is correct"""
        _, r = zdisplacement(self.p)
        self.assertEqual(r, self.p)

    def test_coefficients(self):
        """Test coefficients are correct"""
        H, _ = zdisplacement(self.p)
        self.assertEqual(H, -QuadOperator('q0', 1))


class TestRotation(unittest.TestCase):
    """Tests for rotation function"""
    def setUp(self):
        """Parameters"""
        self.phi = 0.452
        self.hbar = 2.

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, r = rotation(0)
        self.assertEqual(r, 0)

    def test_hermitian(self):
        """Test output is hermitian"""
        H, _ = rotation(self.phi, hbar=self.hbar)
        self.assertTrue(is_hermitian(H))
        self.assertTrue(is_hermitian(get_quad_operator(H)))

    def test_gaussian(self):
        """Test output is gaussian"""
        H, _ = rotation(self.phi, hbar=self.hbar)
        res = get_quad_operator(H, hbar=self.hbar).is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        """Test time parameter is correct"""
        _, r = rotation(self.phi, hbar=self.hbar)
        self.assertEqual(r, self.phi)

    def test_coefficients(self):
        """Test coefficients are correct"""
        H, _ = rotation(self.phi, hbar=self.hbar)
        self.assertEqual(H, -BosonOperator('0^ 0')*self.hbar)

    def test_quad_form(self):
        """Test it has the correct form using quadrature operators"""
        H, _ = rotation(self.phi, mode=1, hbar=self.hbar)
        H = normal_ordered(get_quad_operator(H, hbar=self.hbar), hbar=self.hbar)
        expected = QuadOperator('q1 q1', -0.5)
        expected += QuadOperator('p1 p1', -0.5)
        expected += QuadOperator('', 1)
        self.assertEqual(H, expected)


class TestSqueezing(unittest.TestCase):
    """Tests for squeezing function"""
    def setUp(self):
        """Parameters"""
        self.hbar = 2
        self.r = 0.242
        self.phi = 0.452
        self.H, self.t = squeezing(self.r, self.phi, hbar=self.hbar)

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = squeezing(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        """Test output is hermitian"""
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_quad_operator(self.H)))

    def test_gaussian(self):
        """Test output is gaussian"""
        res = get_quad_operator(self.H).is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        """Test time parameter is correct"""
        self.assertEqual(self.t, self.r)

    def test_quad_form(self):
        """Test it has the correct form using quadrature operators"""
        H, _ = squeezing(2, mode=1)
        H = normal_ordered(get_quad_operator(H, hbar=self.hbar), hbar=self.hbar)
        expected = QuadOperator('q1 p1', -1)
        expected += QuadOperator('', 1j)
        self.assertEqual(H, expected)


class TestQuadraticPhase(unittest.TestCase):
    """Tests for quadratic phase function"""
    def setUp(self):
        """Parameters"""
        self.hbar = 2
        self.s = 0.242
        self.H, self.t = quadratic_phase(self.s)

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = quadratic_phase(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        """Test output is hermitian"""
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_boson_operator(self.H, hbar=self.hbar)))

    def test_gaussian(self):
        """Test output is gaussian"""
        res = self.H.is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        """Test time parameter is correct"""
        self.assertEqual(self.t, self.s)

    def test_boson_form(self):
        """Test bosonic form is correct"""
        H = normal_ordered(get_boson_operator(self.H, hbar=self.hbar))
        expected = BosonOperator('0 0', -0.5)
        expected += BosonOperator('0^ 0', -1)
        expected += BosonOperator('0^ 0^', -0.5)
        expected += BosonOperator('', -0.5)
        self.assertEqual(H, expected)


class TestBeamsplitter(unittest.TestCase):
    """Tests for beamsplitter function"""
    def setUp(self):
        """Parameters"""
        self.hbar = 2
        self.theta = 0.242
        self.phi = 0.452
        self.H, self.t = beamsplitter(self.theta, self.phi, hbar=self.hbar)

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = beamsplitter(0, 0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        """Test output is hermitian"""
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_quad_operator(self.H)))

    def test_gaussian(self):
        """Test output is gaussian"""
        res = get_quad_operator(self.H).is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        """Test time parameter is correct"""
        self.assertEqual(self.t, self.theta)

    def test_quad_form(self):
        """Test it has the correct form using quadrature operators"""
        H, _ = beamsplitter(np.pi/4, np.pi/2, mode1=1, mode2=3, hbar=self.hbar)
        H = normal_ordered(get_quad_operator(H, hbar=self.hbar), hbar=self.hbar)
        expected = QuadOperator('q1 q3', -1)
        expected += QuadOperator('p1 p3', -1)
        self.assertEqual(H, expected)


class TestTwoModeSqueezing(unittest.TestCase):
    """Tests for two-mode squeezing function"""
    def setUp(self):
        """Parameters"""
        self.hbar = 2
        self.r = 0.242
        self.phi = 0.452
        self.H, self.t = two_mode_squeezing(self.r, self.phi, hbar=self.hbar)

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = two_mode_squeezing(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        """Test output is hermitian"""
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_quad_operator(self.H)))

    def test_gaussian(self):
        """Test output is gaussian"""
        res = get_quad_operator(self.H).is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        """Test time parameter is correct"""
        self.assertEqual(self.t, self.r)

    def test_quad_form(self):
        """Test it has the correct form using quadrature operators"""
        H, _ = two_mode_squeezing(2, mode1=1, mode2=3, hbar=self.hbar)
        H = normal_ordered(get_quad_operator(H, hbar=self.hbar), hbar=self.hbar)
        expected = QuadOperator('q1 p3', 1)
        expected += QuadOperator('p1 q3', 1)
        self.assertEqual(H, expected)


class TestControlledAddition(unittest.TestCase):
    """Tests for controlled addition function"""
    def setUp(self):
        """Parameters"""
        self.hbar = 2
        self.s = 0.242
        self.H, self.t = controlled_addition(self.s)

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = controlled_addition(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        """Test output is hermitian"""
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_boson_operator(self.H, hbar=self.hbar)))

    def test_gaussian(self):
        """Test output is gaussian"""
        res = self.H.is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        """Test time parameter is correct"""
        self.assertEqual(self.t, self.s)

    def test_boson_form(self):
        """Test bosonic form is correct"""
        H = normal_ordered(get_boson_operator(self.H, hbar=self.hbar))
        expected = BosonOperator('0 1', -1j)
        expected += BosonOperator('0 1^', 1j)
        expected += BosonOperator('0^ 1', -1j)
        expected += BosonOperator('0^ 1^', 1j)
        self.assertEqual(H, expected)


class TestControlledPhase(unittest.TestCase):
    """Tests for controlled phase function"""
    def setUp(self):
        """Parameters"""
        self.hbar = 2
        self.s = 0.242
        self.H, self.t = controlled_phase(self.s)

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = controlled_phase(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        """Test output is hermitian"""
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_boson_operator(self.H, hbar=self.hbar)))

    def test_gaussian(self):
        """Test output is gaussian"""
        res = self.H.is_gaussian()
        self.assertTrue(res)

    def test_time(self):
        """Test time parameter is correct"""
        self.assertEqual(self.t, self.s)

    def test_boson_form(self):
        """Test bosonic form is correct"""
        H = normal_ordered(get_boson_operator(self.H, hbar=self.hbar))
        expected = BosonOperator('0 1', -1)
        expected += BosonOperator('0 1^', -1)
        expected += BosonOperator('0^ 1', -1)
        expected += BosonOperator('0^ 1^', -1)
        self.assertEqual(H, expected)


class TestCubicPhase(unittest.TestCase):
    """Tests for cubic phase function"""
    def setUp(self):
        """Parameters"""
        self.hbar = 2
        self.gamma = 0.242
        self.H, self.t = cubic_phase(self.gamma)

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = cubic_phase(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        """Test output is hermitian"""
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_boson_operator(self.H, hbar=self.hbar)))

    def test_gaussian(self):
        """Test output is gaussian"""
        res = self.H.is_gaussian()
        self.assertFalse(res)

    def test_time(self):
        """Test time parameter is correct"""
        self.assertEqual(self.t, self.gamma)


class TestKerr(unittest.TestCase):
    """Tests for Kerr function"""
    def setUp(self):
        """Parameters"""
        self.hbar = 2
        self.kappa = 0.242
        self.H, self.t = kerr(self.kappa, hbar=self.hbar)

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = two_mode_squeezing(0)
        self.assertEqual(t, 0)

    def test_hermitian(self):
        """Test output is hermitian"""
        self.assertTrue(is_hermitian(self.H))
        self.assertTrue(is_hermitian(get_quad_operator(self.H)))

    def test_gaussian(self):
        """Test output is gaussian"""
        res = get_quad_operator(self.H).is_gaussian()
        self.assertFalse(res)

    def test_time(self):
        """Test time parameter is correct"""
        self.assertEqual(self.t, self.kappa)


if __name__ == '__main__':
    unittest.main()
