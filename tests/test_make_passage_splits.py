"""Tests for scripts/make_passage_splits.py (loaded by file path, so no package setup is needed).

The synthetic data uses made-up words such as "p3w17": passage 3, word 17. Different
passages never share a word window, and a copy of a passage with other line breaks is
made by cutting the same word sequence at another width.
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "make_passage_splits.py"
SPEC = importlib.util.spec_from_file_location("make_passage_splits", MODULE_PATH)
splits_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(splits_module)

sys.path.insert(0, str(REPO_ROOT))
try:
    from src.common.contract_checks import check_dataset_index_records, check_splits
except ImportError:  # the repository contract checks are not importable here
    check_dataset_index_records = check_splits = None


def passage_words(passage, count):
    return ["p%dw%d" % (passage, n) for n in range(count)]


def cut(words, width):
    return [" ".join(words[i:i + width]) for i in range(0, len(words), width)]


def page_records(page, split, texts):
    return [{"id": "line_%s_%d" % (page, i), "image_path": "%s/%d.png" % (page, i),
             "gt_text": text, "split": split, "group_id": page}
            for i, text in enumerate(texts, 1)]


def build_records():
    """Pages 1 and 2 copy passage 1 (other line breaks); test page 100 is passage 1 too."""
    records = []
    records += page_records("1", "train", cut(passage_words(1, 40), 8))
    records += page_records("2", "train", cut(passage_words(1, 40), 7))
    records += page_records("100", "test", cut(passage_words(1, 40), 8))
    records += page_records("101", "test", cut(passage_words(9, 24), 8))
    for page, passage in (("3", 2), ("4", 2), ("5", 2), ("6", 3), ("7", 3), ("8", 4),
                          ("9", 5), ("10", 6), ("11", 6), ("12", 7), ("13", 8)):
        records.append(None)
        records.pop()
        records += page_records(page, "train", cut(passage_words(passage, 40), 8))
    return records


def run_pipeline(records, link=0.4, val_fraction=0.3, seed=42):
    pages, official = splits_module.group_pages(records)
    clusters = splits_module.find_passages(pages, 5, link)
    roles = splits_module.assign_roles(clusters, pages, official, val_fraction, seed)
    return pages, clusters, roles


class FindPassagesTests(unittest.TestCase):
    def test_copies_with_other_line_breaks_form_one_passage(self):
        pages, official = splits_module.group_pages(build_records())
        clusters = splits_module.find_passages(pages, 5, 0.4)
        self.assertIn(["1", "2", "100"], clusters)
        self.assertIn(["3", "4", "5"], clusters)
        self.assertIn(["101"], clusters)
        self.assertIn(["8"], clusters)

    def test_link_threshold_decides_weak_overlap(self):
        first = page_records("1", "train", [" ".join(passage_words(1, 14))])
        second_words = passage_words(1, 7) + ["x%d" % n for n in range(7)]
        second = page_records("2", "train", [" ".join(second_words)])
        pages, _ = splits_module.group_pages(first + second)
        # 10 windows each, 3 shared: a share of 0.3
        self.assertEqual(len(splits_module.find_passages(pages, 5, 0.4)), 2)
        self.assertEqual(len(splits_module.find_passages(pages, 5, 0.25)), 1)

    def test_links_are_transitive(self):
        # one continuous line per page: pages 1 and 2 overlap, pages 2 and 3 overlap,
        # pages 1 and 3 share a single window only
        words = passage_words(1, 60)
        records = (page_records("1", "train", [" ".join(words[:30])])
                   + page_records("2", "train", [" ".join(words[10:45])])
                   + page_records("3", "train", [" ".join(words[25:60])]))
        pages, _ = splits_module.group_pages(records)
        self.assertEqual(splits_module.find_passages(pages, 5, 0.3), [["1", "2", "3"]])
        self.assertEqual(splits_module.find_passages(pages, 5, 0.9), [["1"], ["2"], ["3"]])

    def test_passages_and_pages_use_natural_order(self):
        records = []
        for page, passage in (("10", 1), ("2", 2), ("33", 3)):
            records += page_records(page, "train", cut(passage_words(passage, 16), 8))
        pages, _ = splits_module.group_pages(records)
        self.assertEqual(splits_module.find_passages(pages, 5, 0.4), [["2"], ["10"], ["33"]])

    def test_page_with_only_short_lines_stays_alone(self):
        records = (page_records("1", "train", ["too short", "also short"])
                   + page_records("2", "train", ["too short", "also short"]))
        pages, _ = splits_module.group_pages(records)
        self.assertEqual(splits_module.find_passages(pages, 5, 0.4), [["1"], ["2"]])


class AssignRolesTests(unittest.TestCase):
    def test_pages_sharing_a_passage_with_test_are_excluded(self):
        _, _, roles = run_pipeline(build_records())
        self.assertEqual(roles["1"], "excluded")
        self.assertEqual(roles["2"], "excluded")
        self.assertEqual(roles["100"], "test")
        self.assertEqual(roles["101"], "test")

    def test_whole_passages_stay_together(self):
        _, clusters, roles = run_pipeline(build_records())
        for cluster in clusters:
            used = {roles[page] for page in cluster if roles[page] != "excluded"}
            self.assertEqual(len(used), 1, cluster)

    def test_val_and_train_are_both_filled(self):
        _, _, roles = run_pipeline(build_records())
        counts = Counter(roles.values())
        self.assertGreater(counts["val"], 0)
        self.assertGreater(counts["train"], 0)

    def test_val_size_stays_near_the_target(self):
        pages, _, roles = run_pipeline(build_records(), val_fraction=0.3)
        pool_lines = sum(len(pages[p]) for p, r in roles.items() if r in ("train", "val"))
        val_lines = sum(len(pages[p]) for p, r in roles.items() if r == "val")
        self.assertLessEqual(val_lines, 0.3 * pool_lines * splits_module.VAL_OVERSHOOT)
        self.assertGreater(val_lines, 0)

    def test_same_seed_gives_same_result_and_seeds_can_differ(self):
        first = run_pipeline(build_records(), seed=7)[2]
        again = run_pipeline(build_records(), seed=7)[2]
        self.assertEqual(first, again)
        variants = {tuple(sorted(p for p, r in run_pipeline(build_records(), seed=s)[2].items()
                                 if r == "val")) for s in range(20)}
        self.assertGreater(len(variants), 1)

    def test_no_official_test_pages_is_an_error(self):
        records = page_records("1", "train", cut(passage_words(1, 16), 8))
        pages, official = splits_module.group_pages(records)
        clusters = splits_module.find_passages(pages, 5, 0.4)
        with self.assertRaisesRegex(ValueError, "no official test pages"):
            splits_module.assign_roles(clusters, pages, official, 0.3, 1)

    def test_val_that_cannot_fit_is_an_error(self):
        records = (page_records("1", "test", cut(passage_words(1, 16), 8))
                   + page_records("2", "train", cut(passage_words(2, 80), 8)))
        pages, official = splits_module.group_pages(records)
        clusters = splits_module.find_passages(pages, 5, 0.4)
        with self.assertRaisesRegex(ValueError, "val"):
            splits_module.assign_roles(clusters, pages, official, 0.1, 1)

    def test_only_one_passage_left_leaves_nothing_for_train(self):
        records = (page_records("1", "test", cut(passage_words(1, 16), 8))
                   + page_records("2", "train", cut(passage_words(2, 16), 8)))
        pages, official = splits_module.group_pages(records)
        clusters = splits_module.find_passages(pages, 5, 0.4)
        with self.assertRaisesRegex(ValueError, "train"):
            splits_module.assign_roles(clusters, pages, official, 0.9, 1)


class ReadAndCheckTests(unittest.TestCase):
    def write_index(self, tmp, records):
        path = os.path.join(tmp, "index.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return path

    def test_missing_field_is_an_error(self):
        record = page_records("1", "train", ["a b c d e f"])[0]
        for key in ("id", "gt_text", "group_id", "split"):
            broken = {k: v for k, v in record.items() if k != key}
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, key):
                    splits_module.read_index(self.write_index(tmp, [broken]))

    def test_duplicate_id_is_an_error(self):
        record = page_records("1", "train", ["a b c d e f"])[0]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                splits_module.read_index(self.write_index(tmp, [record, record]))

    def test_page_split_across_official_splits_is_an_error(self):
        records = (page_records("1", "train", ["a b c d e f"])
                   + [dict(page_records("1", "test", ["g h i j k l"])[0], id="other")])
        with self.assertRaisesRegex(ValueError, "more than one official split"):
            splits_module.group_pages(records)

    def test_check_result_catches_a_passage_in_two_splits(self):
        records = build_records()
        pages, clusters, roles = run_pipeline(records)
        _, splits = splits_module.build_outputs(records, roles)
        self.assertEqual(splits_module.check_result(records, roles, clusters, splits), [])
        broken = dict(roles)
        broken["4"] = "val" if roles["3"] != "val" else "train"
        _, broken_splits = splits_module.build_outputs(records, broken)
        problems = splits_module.check_result(records, broken, clusters, broken_splits)
        self.assertTrue(any("several splits" in problem for problem in problems))

    def test_check_result_catches_changed_test_lines(self):
        records = build_records()
        _, clusters, roles = run_pipeline(records)
        _, splits = splits_module.build_outputs(records, roles)
        splits["line_3_1"] = "test"
        problems = splits_module.check_result(records, roles, clusters, splits)
        self.assertTrue(any("official test" in problem for problem in problems))


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.index = os.path.join(self.tmp.name, "all.jsonl")
        with open(self.index, "w", encoding="utf-8") as handle:
            for record in build_records():
                handle.write(json.dumps(record) + "\n")
        self.out_index = os.path.join(self.tmp.name, "out", "index.jsonl")
        self.out_splits = os.path.join(self.tmp.name, "out", "splits.json")
        self.report = os.path.join(self.tmp.name, "out", "passages.json")

    def run_main(self, extra=()):
        argv = ["--index", self.index, "--out-index", self.out_index,
                "--out-splits", self.out_splits, "--report", self.report,
                "--val-fraction", "0.3"] + list(extra)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = splits_module.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_end_to_end_outputs(self):
        code, out, err = self.run_main()
        self.assertEqual(code, 0, err)
        self.assertIn("excluded (same passage as a test page): 2 pages", out)
        with open(self.out_index, encoding="utf-8") as handle:
            kept = [json.loads(line) for line in handle]
        with open(self.out_splits, encoding="utf-8") as handle:
            splits = json.load(handle)
        with open(self.report, encoding="utf-8") as handle:
            report = json.load(handle)
        all_records = build_records()
        excluded_ids = {r["id"] for r in all_records if r["group_id"] in ("1", "2")}
        self.assertEqual({r["id"] for r in kept}, {r["id"] for r in all_records} - excluded_ids)
        self.assertEqual(set(splits), {r["id"] for r in kept})
        self.assertEqual(report["excluded_pages"], ["1", "2"])
        self.assertEqual(report["parameters"]["seed"], 42)
        self.assertEqual(report["counts"]["pages"]["test"], 2)
        test_ids = {r["id"] for r in all_records if r["split"] == "test"}
        self.assertEqual({i for i, s in splits.items() if s == "test"}, test_ids)
        for record in kept:  # records are copied unchanged, including the official split
            self.assertIn(record, all_records)

    @unittest.skipIf(check_splits is None, "repository contract checks are not importable")
    def test_outputs_pass_the_real_contract_checks(self):
        code, _, err = self.run_main()
        self.assertEqual(code, 0, err)
        with open(self.out_index, encoding="utf-8") as handle:
            kept = [json.loads(line) for line in handle]
        with open(self.out_splits, encoding="utf-8") as handle:
            splits = json.load(handle)
        self.assertEqual(check_dataset_index_records(kept), [])
        self.assertEqual(check_splits(splits, [r["id"] for r in kept]), [])

    def test_two_runs_write_identical_files(self):
        self.assertEqual(self.run_main()[0], 0)
        with open(self.out_splits, encoding="utf-8") as handle:
            first = handle.read()
        self.assertEqual(self.run_main()[0], 0)
        with open(self.out_splits, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), first)

    def test_bad_arguments_return_code_2(self):
        for extra in (["--window", "0"], ["--link", "0"], ["--link", "1.5"],
                      ["--val-fraction", "1"]):
            code, _, err = self.run_main(extra)
            self.assertEqual(code, 2, extra)
            self.assertIn("error:", err)

    def test_output_must_not_overwrite_the_input(self):
        self.out_index = self.index
        code, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("different files", err)

    def test_missing_input_returns_code_2(self):
        self.index = os.path.join(self.tmp.name, "missing.jsonl")
        code, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("error:", err)


if __name__ == "__main__":
    unittest.main()
