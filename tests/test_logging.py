import logging

from app.core.logging import NOISY_LIBRARY_LOGGERS, _configure_noisy_library_loggers


def test_configure_noisy_library_loggers_raises_third_party_loggers_to_warning():
    """高噪声第三方库默认不应该用 INFO 刷屏。"""

    original_levels = {
        logger_name: logging.getLogger(logger_name).level
        for logger_name in NOISY_LIBRARY_LOGGERS
    }

    try:
        for logger_name in NOISY_LIBRARY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.INFO)

        _configure_noisy_library_loggers()

        for logger_name, logger_level in NOISY_LIBRARY_LOGGERS.items():
            assert logging.getLogger(logger_name).level == logger_level
    finally:
        for logger_name, logger_level in original_levels.items():
            logging.getLogger(logger_name).setLevel(logger_level)
