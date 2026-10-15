# SPDX-License-Identifier: Apache-2.0
import os
import sys

# The spot_sim package runs under Isaac Sim's Python; its pure-Python modules are tested here.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "isaac"))
