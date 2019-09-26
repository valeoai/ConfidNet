"""
    Logger utilities. It improves levels of logger and add coloration for each level
"""
import logging
import sys

import coloredlogs
import verboselogs


def get_logger(logger_name, level="SPAM"):
    """
        Get the logger and:
            - improve logger levels:
                    spam, debug, verbose, info, notice, warning, success, error, critical
            - add colors to each levels, and print ms to the time

        Use:
            As a global variable in each file:
         >>>    LOGGER = logger.get_logger('name_of_the_file', level='DEBUG')
            The level allows to reduce the printed messages

        Jupyter notebook:
            Add argument "isatty=True" to coloredlogs.install()
            Easier to read with 'fmt = "%(name)s[%(process)d] %(levelname)s %(message)s"'
    """
    verboselogs.install()
    logger = logging.getLogger(logger_name)

    field_styles = {
        "hostname": {"color": "magenta"},
        "programname": {"color": "cyan"},
        "name": {"color": "blue"},
        "levelname": {"color": "black", "bold": True, "bright": True},
        "asctime": {"color": "green"},
    }
    coloredlogs.install(
        level=level,
        logger=logger,
        fmt="%(asctime)s,%(msecs)03d %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s",
        field_styles=field_styles,
    )

    return logger


def display_progress_bar(cur, total):
    """
        Display progress bar.
    """
    bar_len = 30
    filled_len = cur // (total * bar_len)
    bar_waiter = "=" * filled_len + "." * (bar_len - filled_len)
    sys.stdout.write(f"\r{cur}/{total} [{bar_waiter}] ")
    sys.stdout.flush()
