from pc15core import errors
def test_errors_import():
    assert hasattr(errors, "MissingCudaError")
