import sys


def test_python_version():
    assert sys.version_info >= (3, 11)


def test_package_is_importable():
    import agentops

    assert agentops.__version__ == "0.1.0"
