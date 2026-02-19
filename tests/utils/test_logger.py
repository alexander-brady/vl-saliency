import logging

import vl_saliency.utils.logger as m


def test_default_log_level(monkeypatch):
    monkeypatch.delenv(m.ENV_LOG_LEVEL_KEY, raising=False)
    logger = m.get_logger("test_default")
    assert logger.level == logging.INFO


def test_env_log_level_override(monkeypatch):
    monkeypatch.setenv(m.ENV_LOG_LEVEL_KEY, "debug")
    logger = m.get_logger("test_env")
    assert logger.level == logging.DEBUG


def test_warning_once_logs_once(caplog):
    logger = m.get_logger("test_warning_once")
    with caplog.at_level(logging.WARNING):
        logger.warning_once("hello")
        logger.warning_once("hello")

    assert len(caplog.records) == 1
    assert caplog.records[0].message == "hello"


def test_info_once_logs_once(caplog):
    logger = m.get_logger("test_info_once")
    with caplog.at_level(logging.INFO):
        logger.info_once("hello")
        logger.info_once("hello")

    assert len(caplog.records) == 1
    assert caplog.records[0].message == "hello"
