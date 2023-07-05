############################################
#   Utility Functions
############################################

import logging


class DuplicateFilter(logging.Filter):
    def filter(self, record):
        current_log = (record.module, record.levelno, record.msg)
        return True if current_log == getattr(self, "_last_log", None) else False


def set_logging(logger, default_level=logging.INFO, log_path=""):
    # TODO add log_path and level to config
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    logger.setLevel(default_level)
    # logger.addFilter(DuplicateFilter())
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=(1024**2 * 2), backupCount=3
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logging.addHandler(file_handler)


# TODO add EDA visualization
