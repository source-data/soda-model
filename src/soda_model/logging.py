"""
Sets up the logging infrastructure for all parts of the sd-gprah application.

Everything is logged to stdout by default, each top-level package
(i.e. neotools, neoflask, ...) gets its own log file.
All log files are located in the `log` directory.
"""
import copy
import logging.config
import os
import logging

# Where all log files are stored
LOGGING_BASE_DIR = "./.log"

# The non-duplicated portion Aof the logging configuration. This object is
# (deep-) copied in configure_logging() and
# extended with the very duplicated configuration for each package.
BASE_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console": {"format": "%(asctime)s - %(levelname)s - %(message)s"},
        "simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "console",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {},
    "root": {
        "level": "INFO",
        "handlers": [
            "console",
        ],
    },
}


def configure_logging():
    """Sets up the logging infrastructure: all needed directories & files,
    handlers, loggers, etc.
    """

    # Make sure our logfile directory exists
    if not os.path.exists(LOGGING_BASE_DIR):
        os.mkdir(LOGGING_BASE_DIR)

    # extend the basic config with each package's custom, but very similar
    # configuration.
    config = copy.deepcopy(BASE_CONFIG)
    for module_name in [
        "neoflask",
        "neotools",
        "ontoneo",
        "peerreview",
        "sdg",
        "twitter",
    ]:
        module_handler_name = f"{module_name}_file"
        config["handlers"][module_handler_name] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": f"{LOGGING_BASE_DIR}/{module_name}.log",
            "maxBytes": 10485760,
            "backupCount": 10,
            "encoding": "utf8",
        }
        config["loggers"][module_name] = {
            "level": "DEBUG",
            "handlers": [
                "console",
                module_handler_name,
            ],
            "propagate": False,
        }
    logging.config.dictConfig(config)

    root_logger = get_logger(name=None)
    root_logger.debug("LOGGING CONFIGURED")
    root_logger.debug(config)


def get_logger(*args, **kwargs):
    """
    Returns a logger instance.
    Currently, all arguments are simply passed to the Python standard
    library's logging.getLogger() function, so refer
    to https://docs.python.org/3/library/logging.html#logging.getLogger
    for further information.
    """

    return logging.getLogger(*args, **kwargs)  # type: ignore
