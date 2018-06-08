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

from openfermion.hamiltonians import bose_hubbard

import strawberryfields as sf
from strawberryfields.ops import *

from SFopenfermion.ops import *


class TestBoseHubbardPropagation(unittest.TestCase):
    def setUp(self):
        self.hbar = 2.
        self.eng, _ = sf.Engine(2, hbar=self.hbar)
        self.J = -1
        self.U = 1.5
        self.t = 1.086
        self.k = 20
        self.tol = 1e-2

    def test_1x2(self):
        q = self.eng.register
        H = bose_hubbard(1, 2, self.J, self.U)

        with self.eng:
            Fock(2) | q[0]
            BoseHubbardPropagation(H, self.t, self.k) | q

        state = self.eng.run('fock', cutoff_dim=7)

        Hm = -self.J*np.sqrt(2)*np.array([[0,1,0],[1,0,1],[0,1,0]]) \
            + self.U*np.diag([1,0,1])
        init_state = np.array([1,0,0])
        exp = np.abs(np.dot(expm(-1j*self.t*Hm), init_state))**2

        self.assertTrue(np.allclose(state.fock_prob([2, 0]), exp[0], rtol=self.tol))
        self.assertTrue(np.allclose(state.fock_prob([1, 1]), exp[1], rtol=self.tol))
        self.assertTrue(np.allclose(state.fock_prob([0, 2]), exp[2], rtol=self.tol))


if __name__ == '__main__':
    unittest.main()
