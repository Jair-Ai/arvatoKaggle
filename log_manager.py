import logging
import sys

from .config import settings


class Logging:

    def __init__(self):
        log_level: int = logging.getLevelName(settings.LOG_LEVEL)

        self.root: logging.Logger = logging.getLogger('arvato')
        self.root.setLevel(log_level)

        self.channel: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        self.channel.setLevel(log_level)
        self.formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s [%(process)d]: %(message)s')
        self.channel.setFormatter(self.formatter)
        self.root.addHandler(self.channel)

    def debug(self, text):
        self.root.debug(str(text))

    def info(self, text):
        self.root.info(str(text))

    def warning(self, text):
        self.root.warning(str(text))

    def error(self, text):
        self.root.error(str(text))

    def critical(self, text):
        self.root.critical(str(text))

    def exception(self, text):
        self.root.exception(str(text))


log = Logging()
logger = log
