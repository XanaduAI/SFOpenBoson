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

from openfermion.ops import *
from openfermion.utils import is_hermitian
from openfermion.transforms import *
from openfermion.hamiltonians import bose_hubbard

from SFopenfermion._bose_hubbard_trotter import *


class TestExtractTunneling(unittest.TestCase):
    def test_no_tunneling(self):
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 0, 0)
            extract_tunneling(H)

    def test_too_many_terms(self):
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
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0)
            H -= BosonOperator('5^ 6^')
            H -= BosonOperator('6 5')
            extract_tunneling(H)

    def test_coefficients_differ(self):
        with self.assertRaises(BoseHubbardError):
            H = BosonOperator('0 1^', 0.5)
            H += BosonOperator('0^ 1', 1)
            extract_tunneling(H)

    def test_tunneling_1x1(self):
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(1, 1, 1, 0)
            extract_tunneling(H)

    def test_tunneling_1x2(self):
        H = bose_hubbard(1, 2, 0.5, 0)
        res = extract_tunneling(H)
        expected = [[(0, 1)], 0.5]
        self.assertEqual(res, expected)

    def test_tunneling_2x2(self):
        H = bose_hubbard(2, 2, 0.5, 0)
        res = extract_tunneling(H)
        expected = [[(0, 1), (0, 2), (1, 3), (2, 3)], 0.5]
        self.assertEqual(res, expected)

    def test_tunneling_arbitrary(self):
        H = BosonOperator('0 1^', 0.5) + BosonOperator('0^ 1', 0.5)
        H += BosonOperator('0 2^', 0.5) + BosonOperator('0^ 2', 0.5)
        H += BosonOperator('1 2^', 0.5) + BosonOperator('1^ 2', 0.5)
        res = extract_tunneling(H)
        expected = [[(0, 1), (0, 2), (1, 2)], -0.5]
        self.assertEqual(res, expected)


class TestExtractOnsiteChemical(unittest.TestCase):
    def test_incorrect_ladders(self):
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.1, 0.2)
            H -= BosonOperator('5^ 5^ 5^ 5^')
            H -= BosonOperator('5 5^ 5 5^')
            extract_onsite_chemical(H)

    def test_too_many_terms(self):
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.1, 0.2)
            H -= BosonOperator('0^ 0^ 0^ 0^')
            extract_onsite_chemical(H)

        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.1, 0.2)
            H -= BosonOperator('5^ 5 5^ 5')
            extract_onsite_chemical(H)

    def test_differing_chemical_potential(self):
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.1, 0.2)
            H -= BosonOperator('5^ 5')
            extract_onsite_chemical(H)

    def test_wrong_term_length(self):
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.1, 0.2)
            H -= BosonOperator('5^ 5 5^')
            H += BosonOperator('5^ 5')
            extract_onsite_chemical(H)

    def test_differing_onsite(self):
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.1, 0.2)
            H -= BosonOperator('5^ 5 5^ 5')
            H += BosonOperator('5^ 5')
            extract_onsite_chemical(H)

    def test_only_chemical_potential(self):
        H = bose_hubbard(2, 2, 1, 0, 0.2)
        res = extract_onsite_chemical(H)
        expected = ([], [[0, 1, 2, 3], 0.2])
        self.assertEqual(res, expected)

    def test_only_onsite(self):
        H = bose_hubbard(2, 2, 1, 0.1, 0)
        res = extract_onsite_chemical(H)
        expected = ([[0, 1, 2, 3], 0.1], [])
        self.assertEqual(res, expected)

    def test_both(self):
        H = bose_hubbard(2, 2, 1, 0.1, 0.2)
        res = extract_onsite_chemical(H)
        expected = ([[0, 1, 2, 3], 0.1], [[0, 1, 2, 3], 0.2])
        self.assertEqual(res, expected)


class TestExtractDipole(unittest.TestCase):
    def test_no_dipole(self):
        H = bose_hubbard(2, 2, 1, 1, 1, 0)
        res = extract_dipole(H)
        self.assertEqual(res, [])

    def test_too_many_terms(self):
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 1, 1, 0.5)
            H += BosonOperator('0^ 0 1^ 2', 0.5)
            extract_dipole(H)

    def test_ladder_wrong_form(self):
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 1, 1, 0.5)
            H += BosonOperator('5^ 5 6^ 6^')
            H -= BosonOperator('6 5')
            extract_dipole(H)

    def test_coefficients_differ(self):
        with self.assertRaises(BoseHubbardError):
            H = BosonOperator('0^ 0 1^ 1', 0.5)
            H += BosonOperator('0^ 0 2^ 2', 1)
            extract_dipole(H)

    def test_2x2(self):
        H = bose_hubbard(2, 2, 1, 0, 0, 0.5)
        res = extract_dipole(H)
        expected = [[(0, 1), (0, 2), (1, 3), (2, 3)], 0.5]
        self.assertEqual(res, expected)

    def test_arbitrary(self):
        H = BosonOperator('0^ 0 5^ 5', 0.1)
        res = extract_dipole(H)
        expected = [[(0, 5)], 0.1]
        self.assertEqual(res, expected)


class TestTrotter(unittest.TestCase):
    def setUp(self):
        self.J = 1
        self.U = 0.5
        self.mu = 0.25
        self.t = 1.068
        self.k = 20

    def test_invalid(self):
        with self.assertRaises(BoseHubbardError):
            H = BosonOperator('0')
            res = trotter_layer(H, self.t, self.k)

    def test_dipole(self):
        with self.assertRaises(BoseHubbardError):
            H = bose_hubbard(2, 2, 1, 0.5, 0, 1)
            res = trotter_layer(H, self.t, self.k)

    def test_tunneling_2x2(self):
        H = bose_hubbard(2, 2, self.J, 0, 0)
        res = trotter_layer(H, self.t, self.k)
        theta = -self.t*self.J/self.k
        phi = np.pi/2
        expected = {'BS': (theta, phi, [(0, 1), (0, 2), (1, 3), (2, 3)])}
        self.assertEqual(res, expected)

    def test_onsite_2x2(self):
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
