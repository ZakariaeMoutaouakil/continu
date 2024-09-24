from logging import Formatter, StreamHandler, getLogger, Logger, DEBUG
from typing import Optional


def get_basic_logger(name: str, level: Optional[int] = DEBUG) -> Logger:
    """
    Creates and returns a basic logger that only prints to the console.

    Args:
        name (str): The name of the logger.
        level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured logger instance.

    Raises:
        ValueError: If the provided logging level is not valid.
    """
    if not isinstance(level, int):
        raise ValueError("Logging level must be an integer.")

    logger = getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = StreamHandler()
        console_handler.setLevel(level)
        formatter = Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def main():
    # Example usage
    try:
        logger = get_basic_logger("example_logger")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")

        # Using a custom log level
        debug_logger = get_basic_logger("debug_logger", DEBUG)
        debug_logger.debug("This is a debug message")

        # This will raise a ValueError
        # invalid_logger = get_basic_logger("invalid_logger", "INFO")
    except ValueError as e:
        print(f"ValueError: {e}")


if __name__ == "__main__":
    main()
