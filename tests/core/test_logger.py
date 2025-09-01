import logging

from vl_saliency.core.logger import get_logger


def test_logger_emits_messages_respects_level(capsys):
    logger = get_logger("behave", level=logging.WARNING)

    logger.info("should NOT appear")
    logger.warning("should appear")

    captured = capsys.readouterr()
    assert "should NOT appear" not in captured.out
    assert "should appear" in captured.out


def test_no_duplicate_handlers_on_multiple_calls(capsys):
    logger1 = get_logger("dup")
    logger2 = get_logger("dup")

    assert logger1 is logger2
    assert len(logger1.handlers) == 1  # no duplicates

    logger1.error("error once")
    captured = capsys.readouterr()
    # message should appear only once
    assert captured.out.count("error once") == 1
