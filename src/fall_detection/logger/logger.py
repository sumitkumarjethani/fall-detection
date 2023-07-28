import logging


def configure_logging():
    """
        Configures the logger and the messages to use in different .py
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(module)s - %(funcName)s: %(message)s",
    )

    waitress_logger = logging.getLogger('waitress')
    waitress_logger.setLevel(logging.INFO)

    # create logger
    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.DEBUG)
