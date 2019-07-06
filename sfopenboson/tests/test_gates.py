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
import pytest

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


class TestDisplacement:
    """Tests for Displacement function"""
    alpha = 0.452+0.12j

    def test_identity(self, hbar):
        """Test alpha=0 gives identity"""
        H, r = displacement(0, hbar=hbar)
        assert H == BosonOperator.identity()
        assert r == 0

    def test_hermitian(self, hbar):
        """Test output is hermitian"""
        H, _ = displacement(self.alpha, hbar=hbar)
        assert is_hermitian(H)
        assert is_hermitian(get_quad_operator(H))

    def test_gaussian(self, hbar):
        """Test output is gaussian"""
        H, _ = displacement(self.alpha, hbar=hbar)
        res = get_quad_operator(H, hbar=hbar).is_gaussian()
        assert res

    def test_time(self, hbar):
        """Test time parameter is correct"""
        _, r = displacement(self.alpha)
        assert r == np.abs(self.alpha)

    def test_coefficients(self, hbar):
        """Test coefficients are correct"""
        H, _ = displacement(self.alpha, hbar=hbar)
        phi = np.angle(self.alpha)

        for term, coeff in H.terms.items():
            assert len(term) == 1
            j = 1-term[0][1]
            expected = (-1)**j * np.exp(1j*phi*(-1)**j)*hbar
            assert coeff == 1j*expected


class TestXDisplacement:
    """Tests for x displacement function"""
    x = 0.452

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, r = xdisplacement(0)
        assert r == 0

    def test_hermitian(self, hbar):
        """Test output is hermitian"""
        H, _ = xdisplacement(self.x)
        assert is_hermitian(H)
        assert is_hermitian(get_boson_operator(H, hbar=hbar))

    def test_gaussian(self):
        """Test output is gaussian"""
        H, _ = xdisplacement(self.x)
        res = H.is_gaussian()
        assert res

    def test_time(self):
        """Test time parameter is correct"""
        _, r = xdisplacement(self.x)
        assert r == self.x

    def test_coefficients(self):
        """Test coefficients are correct"""
        H, _ = xdisplacement(self.x)
        assert H == QuadOperator('p0', 1)


class TestZDisplacement:
    """Tests for z displacement function"""
    p = 0.452

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, r = zdisplacement(0)
        assert r == 0

    def test_hermitian(self, hbar):
        """Test output is hermitian"""
        H, _ = zdisplacement(self.p)
        assert is_hermitian(H)
        assert is_hermitian(get_boson_operator(H, hbar=hbar))

    def test_gaussian(self):
        """Test output is gaussian"""
        H, _ = zdisplacement(self.p)
        res = H.is_gaussian()
        assert res

    def test_time(self):
        """Test time parameter is correct"""
        _, r = zdisplacement(self.p)
        assert r == self.p

    def test_coefficients(self):
        """Test coefficients are correct"""
        H, _ = zdisplacement(self.p)
        assert H == -QuadOperator('q0', 1)


class TestRotation:
    """Tests for rotation function"""
    phi = 0.452

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, r = rotation(0)
        assert r == 0

    def test_hermitian(self, hbar):
        """Test output is hermitian"""
        H, _ = rotation(self.phi, hbar=hbar)
        assert is_hermitian(H)
        assert is_hermitian(get_quad_operator(H))

    def test_gaussian(self, hbar):
        """Test output is gaussian"""
        H, _ = rotation(self.phi, hbar=hbar)
        res = get_quad_operator(H, hbar=hbar).is_gaussian()
        assert res

    def test_time(self, hbar):
        """Test time parameter is correct"""
        _, r = rotation(self.phi, hbar=hbar)
        assert r == self.phi

    def test_coefficients(self, hbar):
        """Test coefficients are correct"""
        H, _ = rotation(self.phi, hbar=hbar)
        assert H == -BosonOperator('0^ 0')*hbar

    def test_quad_form(self, hbar):
        """Test it has the correct form using quadrature operators"""
        H, _ = rotation(self.phi, mode=1, hbar=hbar)
        H = normal_ordered(get_quad_operator(H, hbar=hbar), hbar=hbar)
        expected = QuadOperator('q1 q1', -1/hbar)
        expected += QuadOperator('p1 p1', -1/hbar)
        expected += QuadOperator('', 1)
        assert H == expected


class TestSqueezing:
    """Tests for squeezing function"""
    r = 0.242
    phi = 0.452

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = squeezing(0)
        assert t == 0

    def test_hermitian(self, hbar):
        """Test output is hermitian"""
        H, _ = squeezing(self.r, self.phi, hbar=hbar)
        assert is_hermitian(H)
        assert is_hermitian(get_quad_operator(H))

    def test_gaussian(self, hbar):
        """Test output is gaussian"""
        H, _ = squeezing(self.r, self.phi, hbar=hbar)
        res = get_quad_operator(H).is_gaussian()
        assert res

    def test_time(self, hbar):
        """Test time parameter is correct"""
        H, t = squeezing(self.r, self.phi, hbar=hbar)
        assert t == self.r

    def test_quad_form(self, hbar):
        """Test it has the correct form using quadrature operators"""
        H, _ = squeezing(2, mode=1)
        H = normal_ordered(get_quad_operator(H, hbar=hbar), hbar=hbar)
        expected = QuadOperator('q1 p1', -1)
        expected += QuadOperator('', 1j)
        assert H == expected


class TestQuadraticPhase:
    """Tests for quadratic phase function"""
    s = 0.242
    H, t = quadratic_phase(s)

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = quadratic_phase(0)
        assert t == 0

    def test_hermitian(self, hbar):
        """Test output is hermitian"""
        assert is_hermitian(self.H)
        assert is_hermitian(get_boson_operator(self.H, hbar=hbar))

    def test_gaussian(self):
        """Test output is gaussian"""
        res = self.H.is_gaussian()
        assert res

    def test_time(self):
        """Test time parameter is correct"""
        assert self.t == self.s

    def test_boson_form(self, hbar):
        """Test bosonic form is correct"""
        H = normal_ordered(get_boson_operator(self.H, hbar=hbar))
        expected = BosonOperator('0 0', -0.5)
        expected += BosonOperator('0^ 0', -1)
        expected += BosonOperator('0^ 0^', -0.5)
        expected += BosonOperator('', -0.5)
        assert H == expected


class TestBeamsplitter:
    """Tests for beamsplitter function"""
    theta = 0.242
    phi = 0.452

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = beamsplitter(0, 0)
        assert t == 0

    def test_hermitian(self, hbar):
        """Test output is hermitian"""
        H, _ = beamsplitter(self.theta, self.phi, hbar=hbar)
        assert is_hermitian(H)
        assert is_hermitian(get_quad_operator(H))

    def test_gaussian(self, hbar):
        """Test output is gaussian"""
        H, _ = beamsplitter(self.theta, self.phi, hbar=hbar)
        res = get_quad_operator(H).is_gaussian()
        assert res

    def test_time(self, hbar):
        """Test time parameter is correct"""
        _, t = beamsplitter(self.theta, self.phi, hbar=hbar)
        assert t == self.theta

    def test_quad_form(self, hbar):
        """Test it has the correct form using quadrature operators"""
        H, _ = beamsplitter(np.pi/4, np.pi/2, mode1=1, mode2=3, hbar=hbar)
        H = normal_ordered(get_quad_operator(H, hbar=hbar), hbar=hbar)
        expected = QuadOperator('q1 q3', -1)
        expected += QuadOperator('p1 p3', -1)
        assert H == expected


class TestTwoModeSqueezing:
    """Tests for two-mode squeezing function"""
    r = 0.242
    phi = 0.452

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = two_mode_squeezing(0)
        assert t == 0

    def test_hermitian(self, hbar):
        """Test output is hermitian"""
        H, _ = two_mode_squeezing(self.r, self.phi, hbar=hbar)
        assert is_hermitian(H)
        assert is_hermitian(get_quad_operator(H))

    def test_gaussian(self, hbar):
        """Test output is gaussian"""
        H, _ = two_mode_squeezing(self.r, self.phi, hbar=hbar)
        res = get_quad_operator(H).is_gaussian()
        assert res

    def test_time(self, hbar):
        """Test time parameter is correct"""
        _, t = two_mode_squeezing(self.r, self.phi, hbar=hbar)
        assert t == self.r

    def test_quad_form(self, hbar):
        """Test it has the correct form using quadrature operators"""
        H, _ = two_mode_squeezing(2, mode1=1, mode2=3, hbar=hbar)
        H = normal_ordered(get_quad_operator(H, hbar=hbar), hbar=hbar)
        expected = QuadOperator('q1 p3', 1)
        expected += QuadOperator('p1 q3', 1)
        assert H == expected


class TestControlledAddition:
    """Tests for controlled addition function"""
    s = 0.242
    H, t = controlled_addition(s)

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = controlled_addition(0)
        assert t == 0

    def test_hermitian(self, hbar):
        """Test output is hermitian"""
        assert is_hermitian(self.H)
        assert is_hermitian(get_boson_operator(self.H, hbar=hbar))

    def test_gaussian(self):
        """Test output is gaussian"""
        res = self.H.is_gaussian()
        assert res

    def test_time(self):
        """Test time parameter is correct"""
        assert self.t == self.s

    def test_boson_form(self, hbar):
        """Test bosonic form is correct"""
        H = normal_ordered(get_boson_operator(self.H, hbar=hbar))
        expected = BosonOperator('0 1', -1j)
        expected += BosonOperator('0 1^', 1j)
        expected += BosonOperator('0^ 1', -1j)
        expected += BosonOperator('0^ 1^', 1j)
        assert H == expected


class TestControlledPhase:
    """Tests for controlled phase function"""
    s = 0.242
    H, t = controlled_phase(s)

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = controlled_phase(0)
        assert t == 0

    def test_hermitian(self, hbar):
        """Test output is hermitian"""
        assert is_hermitian(self.H)
        assert is_hermitian(get_boson_operator(self.H, hbar=hbar))

    def test_gaussian(self):
        """Test output is gaussian"""
        res = self.H.is_gaussian()
        assert res

    def test_time(self):
        """Test time parameter is correct"""
        assert self.t == self.s

    def test_boson_form(self, hbar):
        """Test bosonic form is correct"""
        H = normal_ordered(get_boson_operator(self.H, hbar=hbar))
        expected = BosonOperator('0 1', -1)
        expected += BosonOperator('0 1^', -1)
        expected += BosonOperator('0^ 1', -1)
        expected += BosonOperator('0^ 1^', -1)
        assert H == expected


class TestCubicPhase:
    """Tests for cubic phase function"""
    gamma = 0.242
    H, t = cubic_phase(gamma)

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = cubic_phase(0)
        assert t == 0

    def test_hermitian(self, hbar):
        """Test output is hermitian"""
        assert is_hermitian(self.H)
        assert is_hermitian(get_boson_operator(self.H, hbar=hbar))

    def test_gaussian(self):
        """Test output is gaussian"""
        res = self.H.is_gaussian()
        assert not res

    def test_time(self):
        """Test time parameter is correct"""
        assert self.t == self.gamma


class TestKerr:
    """Tests for Kerr function"""
    kappa = 0.242

    def test_identity(self):
        """Test alpha=0 gives identity"""
        _, t = kerr(0)
        assert t == 0

    def test_hermitian(self, hbar):
        """Test output is hermitian"""
        H, t = kerr(self.kappa, hbar=hbar)
        assert is_hermitian(H)
        assert is_hermitian(get_quad_operator(H))

    def test_gaussian(self, hbar):
        """Test output is gaussian"""
        H, _ = kerr(self.kappa, hbar=hbar)
        res = get_quad_operator(H).is_gaussian()
        assert not res

    def test_time(self, hbar):
        """Test time parameter is correct"""
        _, t = kerr(self.kappa, hbar=hbar)
        assert t == self.kappa
