#!/usr/bin/env python3
"""Quick env check for T64."""
import sys
sys.path.insert(0, '.')
print(f'Python: {sys.version.split()[0]}')
try:
    from src.robustness import evaluate_robustness, generate_robustness_report
    print('Import OK')
except Exception as e:
    print(f'Import failed: {e}')