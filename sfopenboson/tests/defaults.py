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
"""Test defaults"""
import logging
import os

import unittest

# Set the logging based on an environment variable.
# default logging is set to WARNING.
logLevel = os.environ.get("LOGGING", "WARNING")
numeric_level = getattr(logging, logLevel.upper(), 10)


logging.basicConfig(level=numeric_level, format='\n%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
logging.captureWarnings(True)


class BaseTest(unittest.TestCase):
    """The base unit test class used by SFOpenBoson"""

    def logTestName(self):
        """Log the test method name at the information level"""
        logging.info('%s', self.id())
