import pathlib


def path_to_test_resources() -> pathlib.Path:
    return pathlib.Path(__file__).parent.resolve()
