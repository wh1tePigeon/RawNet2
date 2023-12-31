import json
import unittest

from source.tests.utils import clear_log_folder_after_use
from source.utils.parse_config import ConfigParser


class TestConfig(unittest.TestCase):
    def test_create(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            json.dumps(config_parser.config, indent=2)
