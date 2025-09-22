import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", help="run tests marked slow")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: slow tests that require large models or external deps"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="skipping slow tests; use --runslow to enable")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
