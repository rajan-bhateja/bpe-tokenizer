import logging

def get_logger(name: str) -> logging.Logger:
    """Creates and returns a logger with the specified name."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter with seconds-only timestamps
    formatter = logging.Formatter(
        '%(asctime)s | %(filename)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    logger.addHandler(ch)

    return logger