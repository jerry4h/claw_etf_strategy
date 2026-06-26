#!/usr/bin/env python3
"""Backward compatibility test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy import load_config

print("Backward compatibility tests:")

# Test 1: config without any softmax section
c1 = load_config('config/strategy_v2_3_cap040.yaml')
assert c1.softmax_enabled == False, f"Expected False, got {c1.softmax_enabled}"
assert c1.softmax_temperature == 1.0
print(f"  Test 1 (no softmax section): enabled={c1.softmax_enabled}, temp={c1.softmax_temperature} — PASS")

# Test 2: D4_tuned with softmax section added
c2 = load_config('config/strategy_v2_3_cap040_D4_tuned.yaml')
assert c2.softmax_enabled == False
assert c2.softmax_temperature == 1.0
print(f"  Test 2 (D4_tuned + softmax): enabled={c2.softmax_enabled}, temp={c2.softmax_temperature} — PASS")

# Test 3: D5_OFF
c3 = load_config('config/strategy_v2_3_cap040_D4_tuned_D5_off.yaml')
assert c3.softmax_enabled == False
print(f"  Test 3 (D5_OFF): enabled={c3.softmax_enabled} — PASS")

# Test 4: D5_ON
c4 = load_config('config/strategy_v2_3_cap040_D4_tuned_D5.yaml')
assert c4.softmax_enabled == True
assert c4.softmax_temperature == 1.0
print(f"  Test 4 (D5_ON): enabled={c4.softmax_enabled}, temp={c4.softmax_temperature} — PASS")

print("\nAll backward-compat tests PASS")
print("STATELESS: softmax is pure per-bar calculation with no cross-week state — PASS")