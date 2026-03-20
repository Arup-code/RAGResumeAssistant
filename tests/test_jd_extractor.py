from __future__ import annotations

import unittest
from unittest.mock import patch

from matcher.jd_extractor import JDExtractor
from utils.exceptions import ValidationException


class JDExtractorTests(unittest.TestCase):
    def test_missing_api_key_raises_validation_error(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValidationException):
                JDExtractor(api_key=None)


if __name__ == "__main__":
    unittest.main()



