import sys
import os
import timeit
import logging
from hypothesis import given, strategies as st

# Ensure the parent directory is added to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

from app import AtomicData, FormalTheory, Event, Action, ActionResponse

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

def benchmark_atomic_data():
    # Benchmark creation, encoding, and decoding AtomicData
    setup_code = '''
from app import AtomicData
data_str = AtomicData("test_string")
    '''

    test_code = '''
encoded_data = data_str.encode()
decoded_data = AtomicData("")
decoded_data.decode(encoded_data)
    '''
    
    # Adding globals=globals() to provide access to global variables
    times = timeit.repeat(stmt=test_code, setup=setup_code, repeat=3, number=1000, globals=globals())
    print(f'AtomicData benchmark results: {times}')

def benchmark_formal_theory():
    setup_code = '''
from app import AtomicData, FormalTheory
top_atom = AtomicData(True)
bottom_atom = AtomicData(False)
    '''

    test_code = '''
theory = FormalTheory(top_atom, bottom_atom)
encoded_data = theory.encode()
theory.decode(encoded_data)
    '''
    
    # Adding globals=globals() to provide access to global variables
    times = timeit.repeat(stmt=test_code, setup=setup_code, repeat=3, number=1000, globals=globals())
    print(f'FormalTheory benchmark results: {times}')

@given(st.integers(), st.booleans())
def test_atomic_data_fuzz(int_val, bool_val):
    # Fuzz tests for encoding and decoding integers and booleans
    atomic_int = AtomicData(int_val)
    encoded_int = atomic_int.encode()
    decoded_int = AtomicData(0)
    decoded_int.decode(encoded_int)
    assert decoded_int.value == int_val, f"Failed to decode {int_val}"

    atomic_bool = AtomicData(bool_val)
    encoded_bool = atomic_bool.encode()
    decoded_bool = AtomicData(False)
    decoded_bool.decode(encoded_bool)
    assert decoded_bool.value == bool_val, f"Failed to decode {bool_val}"

@given(st.booleans(), st.booleans())
def test_formal_theory_fuzz(val1, val2):
    top_atom = AtomicData(val1)
    bottom_atom = AtomicData(val2)
    theory = FormalTheory(top_atom, bottom_atom)
    encoded_data = theory.encode()
    theory.decode(encoded_data)
    assert theory.top_atom.value == val1, f"Failed to decode top_atom {val1}"
    assert theory.bottom_atom.value == val2, f"Failed to decode bottom_atom {val2}"

if __name__ == "__main__":
    # Benchmark tests
    benchmark_atomic_data()
    benchmark_formal_theory()

    # Hypothesis fuzz tests
    test_atomic_data_fuzz()
    test_formal_theory_fuzz()