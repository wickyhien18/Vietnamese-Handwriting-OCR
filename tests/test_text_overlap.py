"""Tests for scripts/check_text_overlap.py (loaded by file path, so no package setup is needed)."""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_text_overlap.py"
SPEC = importlib.util.spec_from_file_location("check_text_overlap", MODULE_PATH)
overlap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlap)

PASSAGE = "the minister of natural resources said the plan was ready for review"


def make_line(line_id, split, group, text):
    return overlap.Line(line_id, split, group, overlap.normalize_tokens(text))


def results_for(train, targets, size=5):
    keys, gram_groups = overlap.build_train_reference(train, size)
    return overlap.compare_with_train(targets, keys, gram_groups, size)


class NormalizeTests(unittest.TestCase):
    def test_case_and_punctuation_spacing_do_not_matter(self):
        spaced = overlap.normalize_tokens('He said " hello there " to them')
        tight = overlap.normalize_tokens('he said "hello there" to them.')
        self.assertEqual(spaced, tight)
        self.assertEqual(spaced, ["he", "said", "hello", "there", "to", "them"])

    def test_unicode_forms_are_unified(self):
        self.assertEqual(overlap.normalize_tokens("cafe\u0301"), overlap.normalize_tokens("caf\u00e9"))

    def test_windows(self):
        tokens = ["a", "b", "c", "d", "e", "f"]
        self.assertEqual(overlap.windows(tokens, 5), ["a b c d e", "b c d e f"])
        self.assertEqual(overlap.windows(tokens[:4], 5), [])


class CompareWithTrainTests(unittest.TestCase):
    def test_exact_match_ignores_punctuation_and_case(self):
        train = [make_line("t1", "train", "1", "One, two three four five six.")]
        target = [make_line("v1", "val", "9", "one two three four five six")]
        result = results_for(train, target)[0]
        self.assertTrue(result.exact)
        self.assertEqual(result.windows, 2)
        self.assertEqual(result.matched, 2)

    def test_same_passage_with_other_line_breaks_is_not_exact_but_overlaps(self):
        train = [make_line("t1", "train", "1", PASSAGE)]
        target = [make_line("v1", "val", "9", "natural resources said the plan was ready for review today")]
        result = results_for(train, target)[0]
        self.assertFalse(result.exact)
        self.assertGreater(result.matched, 0)
        stats = overlap.summarize([result])
        self.assertEqual(stats["exact"], 0)
        self.assertEqual(stats["any"], 1)
        self.assertEqual(stats["major"], 1)

    def test_unrelated_line_has_no_overlap(self):
        train = [make_line("t1", "train", "1", PASSAGE)]
        target = [make_line("v1", "val", "9", "completely different words appear in this other line")]
        stats = overlap.summarize(results_for(train, target))
        self.assertEqual((stats["exact"], stats["any"], stats["major"]), (0, 0, 0))

    def test_short_lines_count_only_for_exact(self):
        train = [make_line("t1", "train", "1", "too short")]
        target = [make_line("v1", "val", "9", "too short")]
        result = results_for(train, target)[0]
        self.assertTrue(result.exact)
        self.assertEqual(result.windows, 0)
        stats = overlap.summarize([result])
        self.assertEqual(stats["lines"], 1)
        self.assertEqual(stats["eligible"], 0)
        self.assertEqual(stats["exact"], 1)
        self.assertEqual(stats["any_pct"], 0.0)
        self.assertEqual(stats["major_pct"], 0.0)

    def test_major_needs_half_of_the_windows(self):
        train = [make_line("t1", "train", "1", "a1 b1 c1 d1 e1 f1")]
        # 6 words -> 2 windows; the first window is shared and the second is not
        half = [make_line("v1", "val", "9", "a1 b1 c1 d1 e1 x1")]
        # 7 words -> 3 windows; only one is shared (a third), so not "major"
        third = [make_line("v2", "val", "9", "a1 b1 c1 d1 e1 x1 y1")]
        self.assertEqual(overlap.summarize(results_for(train, half))["major"], 1)
        self.assertEqual(overlap.summarize(results_for(train, third))["major"], 0)
        self.assertEqual(overlap.summarize(results_for(train, third))["any"], 1)

    def test_group_rows_name_the_top_source_page(self):
        train = [
            make_line("t1", "train", "1", "alpha beta gamma delta epsilon zeta eta"),
            make_line("t2", "train", "2", PASSAGE),
        ]
        targets = [
            make_line("v1", "val", "9", PASSAGE),
            make_line("v2", "val", "9", "the minister of natural resources said the plan"),
            make_line("v3", "val", "8", "nothing here matches anything at all in train"),
        ]
        rows = overlap.group_rows(results_for(train, targets), top=10)
        self.assertEqual([row["group"] for row in rows], ["9", "8"])
        self.assertEqual(rows[0]["top_source"][0], "2")
        self.assertIsNone(rows[1]["top_source"])
        self.assertEqual(rows[0]["stats"]["lines"], 2)

    def test_group_rows_respects_top(self):
        train = [make_line("t1", "train", "1", PASSAGE)]
        targets = [make_line("v%d" % i, "val", str(i), PASSAGE) for i in range(5)]
        self.assertEqual(len(overlap.group_rows(results_for(train, targets), top=3)), 3)


class VerdictTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(overlap.verdict(0.0, "val")[0], "keep")
        self.assertEqual(overlap.verdict(9.99, "val")[0], "keep")
        self.assertEqual(overlap.verdict(10.0, "val")[0], "report")
        self.assertEqual(overlap.verdict(25.0, "val")[0], "report")
        self.assertEqual(overlap.verdict(25.01, "val")[0], "resplit")

    def test_test_split_is_never_told_to_resplit_itself(self):
        level, action = overlap.verdict(60.0, "test")
        self.assertEqual(level, "resplit")
        self.assertIn("fixed", action)
        self.assertIn("re-split val", overlap.verdict(60.0, "val")[1])


class CorpusTests(unittest.TestCase):
    def write(self, directory, name, text, encoding="utf-8"):
        path = os.path.join(directory, name)
        with open(path, "w", encoding=encoding) as handle:
            handle.write(text)
        return path

    def test_corpus_matches_exact_and_windows(self):
        targets = [
            make_line("v1", "val", "9", "one two three four five six"),
            make_line("v2", "val", "9", "red green blue yellow orange purple"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            corpus = self.write(
                tmp, "corpus.txt",
                "\n\nOne, two three four five six.\nred green blue yellow orange pink\n",
                encoding="utf-8-sig")
            results, corpus_lines = overlap.compare_with_corpus(targets, corpus, 5)
        self.assertEqual(corpus_lines, 2)
        self.assertTrue(results[0].exact)
        self.assertEqual(results[0].matched, results[0].windows)
        self.assertFalse(results[1].exact)
        self.assertEqual(results[1].matched, 1)
        self.assertEqual(results[1].windows, 2)


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def files(self, records, splits):
        index = os.path.join(self.tmp.name, "index.jsonl")
        with open(index, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        splits_path = os.path.join(self.tmp.name, "splits.json")
        with open(splits_path, "w", encoding="utf-8") as handle:
            json.dump(splits, handle)
        return index, splits_path

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = overlap.main(argv)
        return code, out.getvalue(), err.getvalue()

    def sample(self):
        records = [
            {"id": "t1", "gt_text": PASSAGE, "group_id": "1"},
            {"id": "t2", "gt_text": "a quite different sentence about another subject entirely", "group_id": "2"},
            {"id": "v1", "gt_text": PASSAGE, "group_id": "8"},
            {"id": "e1", "gt_text": "something else again about unrelated matters of state", "group_id": "9"},
            {"id": "x1", "gt_text": "this id is not in the splits file at all", "group_id": "7"},
        ]
        splits = {"t1": "train", "t2": "train", "v1": "val", "e1": "test", "ghost": "val"}
        return self.files(records, splits)

    def test_end_to_end_report(self):
        index, splits = self.sample()
        code, out, err = self.run_main(["--index", index, "--splits", splits])
        self.assertEqual(code, 0, err)
        self.assertIn("verdict val: RESPLIT", out)
        self.assertIn("verdict test: KEEP", out)
        self.assertIn("1 index ids are not in the splits file", out)
        self.assertIn("1 split ids are not in the index", out)
        self.assertIn("most overlapping val pages", out)

    def test_end_to_end_with_corpus(self):
        index, splits = self.sample()
        corpus = os.path.join(self.tmp.name, "corpus.txt")
        with open(corpus, "w", encoding="utf-8") as handle:
            handle.write(PASSAGE + "\n")
        code, out, err = self.run_main(["--index", index, "--splits", splits, "--corpus", corpus])
        self.assertEqual(code, 0, err)
        self.assertIn("against the corpus", out)
        self.assertIn("1 val/test lines occur word for word in the corpus", out)

    def test_empty_val_does_not_crash(self):
        index, splits = self.files(
            [{"id": "t1", "gt_text": PASSAGE, "group_id": "1"},
             {"id": "e1", "gt_text": PASSAGE, "group_id": "2"}],
            {"t1": "train", "e1": "test"})
        code, out, err = self.run_main(["--index", index, "--splits", splits])
        self.assertEqual(code, 0, err)
        self.assertIn("no val lines, skipped", out)
        self.assertIn("verdict test: RESPLIT", out)

    def test_errors_return_code_2(self):
        index, splits = self.sample()
        missing = os.path.join(self.tmp.name, "nope.json")
        for argv in (["--index", index, "--splits", missing],
                     ["--index", index, "--splits", splits, "--corpus", missing],
                     ["--index", index, "--splits", splits, "--window", "0"]):
            code, out, err = self.run_main(argv)
            self.assertEqual(code, 2)
            self.assertIn("error:", err)

    def test_bad_split_value_and_missing_train(self):
        index, splits = self.files([{"id": "v1", "gt_text": PASSAGE}], {"v1": "validation"})
        code, _, err = self.run_main(["--index", index, "--splits", splits])
        self.assertEqual(code, 2)
        self.assertIn("unknown split values", err)
        index, splits = self.files([{"id": "v1", "gt_text": PASSAGE}], {"v1": "val"})
        code, _, err = self.run_main(["--index", index, "--splits", splits])
        self.assertEqual(code, 2)
        self.assertIn("no training lines", err)


if __name__ == "__main__":
    unittest.main()
