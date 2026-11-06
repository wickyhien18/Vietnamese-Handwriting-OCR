import os
import unittest

from scripts.validate_ocr_out import validate_file
from src.stubs.identity_corrector import correct


class ContractTest(unittest.TestCase):
    def test_sample_file_is_valid(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "contracts", "examples", "ocr_out.sample.jsonl")
        self.assertEqual(validate_file(path), [])

    def test_corrector_contract(self):
        out = correct("a b c")
        self.assertEqual(set(out.keys()), {"text", "conf"})
        self.assertEqual(len(out["conf"]), len(out["text"].split()))


if __name__ == "__main__":
    unittest.main()
