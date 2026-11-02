"""Общая конфигурация тестов: сетевые тесты отделены от обычного прогона."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="запустить тесты с маркером network (живые внешние сервисы)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="network-тест: запускается только с --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
