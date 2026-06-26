try:
    import pandas
    import numpy
    print(f'pandas {pandas.__version__} OK')
    print(f'numpy {numpy.__version__} OK')
except ImportError as e:
    print(f'Missing: {e}')
