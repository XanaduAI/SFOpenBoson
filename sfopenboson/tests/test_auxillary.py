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
"""Tests for auxillary functions"""

import unittest

import numpy as np

from openfermion.ops import BosonOperator, QuadOperator
from openfermion.transforms import get_quad_operator
from openfermion.hamiltonians import bose_hubbard

from sfopenboson.auxillary import (BoseHubbardError,
                                   quadratic_coefficients,
                                   extract_tunneling,
                                   extract_onsite_chemical,
                                   extract_dipole,
                                   trotter_layer)

from sfopenboson.hamiltonians import (displacement,
                                      xdisplacement,
                                      zdisplacement,
                                      rotation,
                                      squeezing,
                                      quadratic_phase,
                                      beamsplitter,
                                      two_mode_squeezing,
                                      controlled_addition,
                                      controlled_phase)


class TestQuadraticCoefficients(unittest.TestCase):
    """Tests for quadratic_coefficients"""
    def setUp(self):
        """set up"""
        self.hbar = 2

    def test_non_hermitian(self):
        """Test exception with non-Hermitian Hamiltonian"""
        with self.assertRaisesRegex(ValueError, "Hamiltonian must be Hermitian"):
            quadratic_coefficients(QuadOperator('q0 p0'))

    def test_non_gaussian(self):
        """Test exception with non-Gaussian Hamiltonian"""
        with self.assertRaisesRegex(ValueError, "Hamiltonian must be Gaussian"):
            quadratic_coefficients(QuadOperator('q0 p0 p1'))

    def test_displacement_vector(self):
        """Test displacement vector extracted"""
        H = QuadOperator('q0', -0.432) + QuadOperator('p0', 3.213)
        A, d = quadratic_coefficients(H)
        expected_A = np.zeros([2, 2])
        expected_d = np.array([3.213, 0.432])
        self.assertTrue(np.allclose(A, expected_A))
        self.assertTrue(np.allclose(d, expected_d))

        _, d = quadratic_coefficients(QuadOperator('q0 q0'))
        expected = np.array([0, 0])
        self.assertTrue(np.allclose(d, expected))

        _, d = quadratic_coefficients(QuadOperator('p0 p1'))
        expected = np.array([0, 0, 0, 0])
        self.assertTrue(np.allclose(d, expected))

    def test_Dgate_displacement(self):
        """Test displacement vector matches Dgate"""
        a = 0.23-0.432j
        H, t = displacement(a, hbar=self.hbar)
        _, d = quadratic_coefficients(get_quad_operator(H, self.hbar))
        expected = np.array([a.real, a.imag])*np.sqrt(2*self.hbar)/t
        self.assertTrue(np.allclose(d, expected))
        self.assertTrue(np.allclose(a, t*(d[0]+d[1]*1j)/np.sqrt(2*self.hbar)))

    def test_Xgate_displacement(self):
        """Test displacement vector matches Xgate"""
        x = 0.1234
        H, _ = xdisplacement(x)
        _, d = quadratic_coefficients(H)
        expected = np.array([1, 0])
        self.assertTrue(np.allclose(d, expected))

    def test_Zgate_displacement(self):
        """Test displacement vector matches Zgate"""
        z = 0.654
        H, _ = zdisplacement(z)
        _, d = quadratic_coefficients(H)
        expected = np.array([0, 1])
        self.assertTrue(np.allclose(d, expected))

    def test_rotation_coeff(self):
        """Test quadratic coefficients for Rgate"""
        # one mode
        H, _ = rotation(0.23, hbar=self.hbar)
        res, d = quadratic_coefficients(get_quad_operator(H, self.hbar))
        expected = -np.diag(np.array([1, 1]))
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([2])))

        # two modes
        H, _ = rotation(0.23, mode=1, hbar=self.hbar)
        res, d = quadratic_coefficients(get_quad_operator(H, self.hbar))
        expected = np.zeros([4, 4])
        expected[1, 1] = -1
        expected[3, 3] = -1
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))

    def test_squeeze_coeff(self):
        """Test quadratic coefficients for Sgate"""
        # one mode
        # pylint: disable=invalid-unary-operand-type
        H, _ = squeezing(0.23, hbar=self.hbar)
        res, d = quadratic_coefficients(get_quad_operator(H, self.hbar))
        expected = -np.array([[0, 1], [1, 0]])
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([2])))

        # two modes
        H, _ = squeezing(0.23, mode=1, hbar=self.hbar)
        res, d = quadratic_coefficients(get_quad_operator(H, self.hbar))
        expected = np.zeros([4, 4])
        expected[1, 3] = -1
        expected[3, 1] = -1
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))

    def test_quadratic_phase_coeff(self):
        """Test quadratic coefficients for Pgate"""
        # one mode
        H, _ = quadratic_phase(0.23)
        res, d = quadratic_coefficients(H)
        expected = np.zeros([2, 2])
        expected[0, 0] = -1
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([2])))

        # two modes
        H, _ = quadratic_phase(0.23, mode=1)
        res, d = quadratic_coefficients(H)
        expected = np.zeros([4, 4])
        expected[1, 1] = -1
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))

    def test_beamsplitter_coeff(self):
        """Test quadratic coefficients for BSgate"""
        # arbitrary beamsplitter
        theta = 0.5423
        phi = 0.3242
        H, _ = beamsplitter(theta, phi, hbar=self.hbar)
        res, d = quadratic_coefficients(get_quad_operator(H, hbar=self.hbar))
        expected = np.zeros([4, 4])
        expected[0, 3] = expected[3, 0] = -np.cos(np.pi-phi)
        expected[1, 2] = expected[2, 1] = np.cos(np.pi-phi)
        expected[0, 1] = expected[1, 0] = -np.sin(np.pi-phi)
        expected[2, 3] = expected[3, 2] = -np.sin(np.pi-phi)
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))

    def test_two_mode_squeeze_coeff(self):
        """Test quadratic coefficients for S2gate"""
        H, _ = two_mode_squeezing(0.23, hbar=self.hbar)
        res, d = quadratic_coefficients(get_quad_operator(H, self.hbar))
        expected = np.fliplr(np.diag([1]*4))
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))

    def test_controlled_addition(self):
        """Test quadratic coefficients for CXgate"""
        H, _ = controlled_addition(0.23)
        res, d = quadratic_coefficients(H)
        expected = np.fliplr(np.diag([1, 0, 0, 1]))
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))

    def test_controlled_phase(self):
        """Test quadratic coefficients for CZgate"""
        H, _ = controlled_phase(0.23)
        res, d = quadratic_coefficients(H)
        expected = np.zeros([4, 4])
        expected[0, 1] = expected[1, 0] = -1
        self.assertTrue(np.allclose(res, expected))
        self.assertTrue(np.allclose(d, np.zeros([4])))


class TestExtractTunneling(unittest.TestCase):
    """Test Bose-Hubbard tunnelling extracted from BosonOperator"""
    def test_no_tunneling(self):
        """Test exception raised for no tunneling"""
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 0, 0)
            extract_tunneling(H)

    def test_too_many_terms(self):
        """Test exception raised for wrong number of terms"""
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0)
            H -= BosonOperator('0^ 1^')
            extract_tunneling(H)

        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0)
            H -= BosonOperator('0^ 1')
            H -= BosonOperator('1^ 0')
            extract_tunneling(H)

    def test_ladder_wrong_form(self):
        """Test exception raised for wrong ladder operators"""
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0)
            H -= BosonOperator('5^ 6^')
            H -= BosonOperator('6 5')
            extract_tunneling(H)

    def test_coefficients_differ(self):
        """Test exception raised for differing coefficients"""
        with self.assertRaises(BoseHubbardError):
            H = BosonOperator('0 1^', 0.5)
            H += BosonOperator('0^ 1', 1)
            extract_tunneling(H)

    def test_tunneling_1x1(self):
        """Test exception raised 1x1 grid"""
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(1, 1, 1, 0)
            extract_tunneling(H)

    def test_tunneling_1x2(self):
        """Test extracted tunneling on 1x2 grid"""
        H = bose_hubbard(1, 2, 0.5, 0)
        res = extract_tunneling(H)
        expected = [[(0, 1)], 0.5]
        self.assertEqual(res, expected)

    def test_tunneling_2x2(self):
        """Test extracted tunneling on 2x2 grid"""
        H = bose_hubbard(2, 2, 0.5, 0)
        res = extract_tunneling(H)
        expected = [[(0, 1), (0, 2), (1, 3), (2, 3)], 0.5]
        self.assertEqual(res, expected)

    def test_tunneling_arbitrary(self):
        """Test extracted tunneling on arbitrary grid"""
        H = BosonOperator('0 1^', 0.5) + BosonOperator('0^ 1', 0.5)
        H += BosonOperator('0 2^', 0.5) + BosonOperator('0^ 2', 0.5)
        H += BosonOperator('1 2^', 0.5) + BosonOperator('1^ 2', 0.5)
        res = extract_tunneling(H)
        expected = [[(0, 1), (0, 2), (1, 2)], -0.5]
        self.assertEqual(res, expected)


class TestExtractOnsiteChemical(unittest.TestCase):
    """Test Bose-Hubbard onsite interactions extracted from BosonOperator"""
    def test_incorrect_ladders(self):
        """Test exception raised for wrong ladder operators"""
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.1, 0.2)
            H -= BosonOperator('5^ 5^ 5^ 5^')
            H -= BosonOperator('5 5^ 5 5^')
            extract_onsite_chemical(H)

    def test_too_many_terms(self):
        """Test exception raised for wrong number of terms"""
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.1, 0.2)
            H -= BosonOperator('0^ 0^ 0^ 0^')
            extract_onsite_chemical(H)

        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.1, 0.2)
            H -= BosonOperator('5^ 5 5^ 5')
            extract_onsite_chemical(H)

    def test_differing_chemical_potential(self):
        """Test exception raised for differing coefficients"""
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.1, 0.2)
            H -= BosonOperator('5^ 5')
            extract_onsite_chemical(H)

    def test_wrong_term_length(self):
        """Test exception raised for wrong terms"""
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.1, 0.2)
            H -= BosonOperator('5^ 5 5^')
            H += BosonOperator('5^ 5')
            extract_onsite_chemical(H)

    def test_differing_onsite(self):
        """Test exception raised for differing coefficients"""
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.1, 0.2)
            H -= BosonOperator('5^ 5 5^ 5')
            H += BosonOperator('5^ 5')
            extract_onsite_chemical(H)

    def test_only_chemical_potential(self):
        """Test case where there is non-zero mu and zero U"""
        H = bose_hubbard(2, 2, 1, 0, 0.2)
        res = extract_onsite_chemical(H)
        expected = ([], [[0, 1, 2, 3], 0.2])
        self.assertEqual(res, expected)

    def test_only_onsite(self):
        """Test case where there is zero mu and non-zero U"""
        H = bose_hubbard(2, 2, 1, 0.1, 0)
        res = extract_onsite_chemical(H)
        expected = ([[0, 1, 2, 3], 0.1], [])
        self.assertEqual(res, expected)

    def test_both(self):
        """Test case where there is non-zero mu and non-zero U"""
        H = bose_hubbard(2, 2, 1, 0.1, 0.2)
        res = extract_onsite_chemical(H)
        expected = ([[0, 1, 2, 3], 0.1], [[0, 1, 2, 3], 0.2])
        self.assertEqual(res, expected)


class TestExtractDipole(unittest.TestCase):
    """Test Bose-Hubbard dipoles extracted from BosonOperator"""
    def test_no_dipole(self):
        """Test case where there is zero V"""
        H = bose_hubbard(2, 2, 1, 1, 1, 0)
        res = extract_dipole(H)
        self.assertEqual(res, [])

    def test_too_many_terms(self):
        """Test exception raised for wrong number of terms"""
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 1, 1, 0.5)
            H += BosonOperator('0^ 0 1^ 2', 0.5)
            extract_dipole(H)

    def test_ladder_wrong_form(self):
        """Test exception raised for wrong ladder operators"""
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 1, 1, 0.5)
            H += BosonOperator('5^ 5 6^ 6^')
            H -= BosonOperator('6 5')
            extract_dipole(H)

    def test_coefficients_differ(self):
        """Test exception raised for differing coefficients"""
        with self.assertRaises(BoseHubbardError):
            H = BosonOperator('0^ 0 1^ 1', 0.5)
            H += BosonOperator('0^ 0 2^ 2', 1)
            extract_dipole(H)

    def test_2x2(self):
        """Test extracted dipole terms on 2x2 grid"""
        H = bose_hubbard(2, 2, 1, 0, 0, 0.5)
        res = extract_dipole(H)
        expected = [[(0, 1), (0, 2), (1, 3), (2, 3)], 0.5]
        self.assertEqual(res, expected)

    def test_arbitrary(self):
        """Test extracted dipole terms on arbitrary grid"""
        H = BosonOperator('0^ 0 5^ 5', 0.1)
        res = extract_dipole(H)
        expected = [[(0, 5)], 0.1]
        self.assertEqual(res, expected)


class TestTrotter(unittest.TestCase):
    """Test Bose-Hubbard trotter layers extracted from BosonOperator"""
    def setUp(self):
        """Test parameters"""
        self.J = 1
        self.U = 0.5
        self.mu = 0.25
        self.t = 1.068
        self.k = 20

    def test_invalid(self):
        """Test exception raised for non-Bose-Hubbard Hamiltonian"""
        with self.assertRaises(BoseHubbardError):
            H = BosonOperator('0')
            _ = trotter_layer(H, self.t, self.k)

    def test_dipole(self):
        """Test exception raised for non-zero dipole terms"""
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.5, 0, 1)
            _ = trotter_layer(H, self.t, self.k)

    def test_tunneling_2x2(self):
        """Test non-interacting 2x2 grid"""
        H = bose_hubbard(2, 2, self.J, 0, 0)
        res = trotter_layer(H, self.t, self.k)
        theta = -self.t*self.J/self.k
        phi = np.pi/2
        expected = {'BS': (theta, phi, [(0, 1), (0, 2), (1, 3), (2, 3)])}
        self.assertEqual(res, expected)

    def test_onsite_2x2(self):
        """Test on-site interacting 2x2 grid"""
        H = bose_hubbard(2, 2, self.J, self.U, 0)
        res = trotter_layer(H, self.t, self.k)
        theta = -self.t*self.J/self.k
        phi = np.pi/2
        kappa = -self.t*self.U/(2*self.k)
        r = -kappa
        expected = {
            'BS': (theta, phi, [(0, 1), (0, 2), (1, 3), (2, 3)]),
            'K': (kappa, [0, 1, 2, 3]),
            'R': (r, [0, 1, 2, 3]),
        }
        self.assertEqual(res, expected)

    def test_chemical_potential_2x2(self):
        """Test on-site interacting and chemical potential on a 2x2 grid"""
        H = bose_hubbard(2, 2, self.J, self.U, self.mu)
        res = trotter_layer(H, self.t, self.k)
        theta = -self.t*self.J/self.k
        phi = np.pi/2
        kappa = -self.t*self.U/(2*self.k)
        r = self.t*(0.5*self.U+self.mu)/(2*self.k)
        expected = {
            'BS': (theta, phi, [(0, 1), (0, 2), (1, 3), (2, 3)]),
            'K': (kappa, [0, 1, 2, 3]),
            'R': (r, [0, 1, 2, 3]),
        }
        self.assertEqual(res, expected)


    def test_arbitrary(self):
        """Test on-site interacting and chemical potential on a 3-cycle"""
        H = BosonOperator('0 1^', -self.J) + BosonOperator('0^ 1', -self.J)
        H += BosonOperator('0 2^', -self.J) + BosonOperator('0^ 2', -self.J)
        H += BosonOperator('1 2^', -self.J) + BosonOperator('1^ 2', -self.J)

        res = trotter_layer(H, self.t, self.k)
        theta = -self.t*self.J/self.k
        phi = np.pi/2
        expected = {'BS': (theta, phi, [(0, 1), (0, 2), (1, 2)])}
        self.assertEqual(res, expected)

        H += BosonOperator('0^ 0 0^ 0', 0.5*self.U) - BosonOperator('0^ 0', 0.5*self.U)
        H += BosonOperator('1^ 1 1^ 1', 0.5*self.U) - BosonOperator('1^ 1', 0.5*self.U)
        H += BosonOperator('2^ 2 2^ 2', 0.5*self.U) - BosonOperator('2^ 2', 0.5*self.U)

        kappa = -self.t*self.U/(2*self.k)
        r = -kappa
        expected = {
            'BS': (theta, phi, [(0, 1), (0, 2), (1, 2)]),
            'K': (kappa, [0, 1, 2]),
            'R': (r, [0, 1, 2]),
        }
        res = trotter_layer(H, self.t, self.k)
        self.assertEqual(res, expected)


if __name__ == '__main__':
    unittest.main()
