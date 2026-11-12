import json
import os
import tempfile
import unittest

from scripts.build_index import IndexBuildError, build_index

TEXT_A = "nhi\u1ec1u b\u1ea1n \u0111\u1ecdc"
TEXT_B = "s\u1ef1 th\u1eadt \u0111\u01b0\u1ee3c n\u00f3i"


def touch(root, rel, content=b"x"):
    path = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(content)


def write_text(root, rel, text):
    touch(root, rel, text.encode("utf-8"))


class BuildIndexTest(unittest.TestCase):
    def test_split_folders_with_tab_label_files(self):
        with tempfile.TemporaryDirectory() as root:
            for split in ("train", "test"):
                touch(root, "%s/lines/a_0001.png" % split)
                touch(root, "%s/lines/a_0002.png" % split)
                write_text(root, "%s/labels.txt" % split, "a_0001.png\t%s\na_0002.png\t%s\n" % (TEXT_A, TEXT_B))
            records, report = build_index(root)
        self.assertEqual(len(records), 4)
        self.assertEqual(report["split_counts"], {"train": 2, "test": 2})
        self.assertEqual({r["id"] for r in records}, {"train_lines_a_0001", "train_lines_a_0002",
                                                      "test_lines_a_0001", "test_lines_a_0002"})
        self.assertTrue(all(r["gt_text"] in (TEXT_A, TEXT_B) for r in records))

    def test_single_json_dict_no_splits(self):
        with tempfile.TemporaryDirectory() as root:
            touch(root, "images/a.png")
            touch(root, "images/b.png")
            write_text(root, "labels.json", json.dumps({"a.png": TEXT_A, "b.png": TEXT_B}, ensure_ascii=False))
            records, report = build_index(root)
        self.assertEqual(len(records), 2)
        self.assertEqual(report["split_counts"], {"none": 2})
        self.assertNotIn("split", records[0])

    def test_json_with_paths_and_nested_text_field(self):
        with tempfile.TemporaryDirectory() as root:
            touch(root, "train/a.png")
            touch(root, "test/a.png")
            data = {"train/a.png": {"text": TEXT_A}, "test/a.png": {"text": TEXT_B}}
            write_text(root, "all.json", json.dumps(data, ensure_ascii=False))
            records, _ = build_index(root)
        by_id = {r["id"]: r for r in records}
        self.assertEqual(by_id["train_a"]["gt_text"], TEXT_A)
        self.assertEqual(by_id["test_a"]["gt_text"], TEXT_B)
        self.assertEqual(by_id["train_a"]["split"], "train")

    def test_json_list_of_objects(self):
        with tempfile.TemporaryDirectory() as root:
            touch(root, "a.png")
            touch(root, "b.png")
            data = [{"filename": "a.png", "label": TEXT_A}, {"filename": "b.png", "label": TEXT_B}]
            write_text(root, "meta.json", json.dumps(data, ensure_ascii=False))
            records, _ = build_index(root)
        self.assertEqual(len(records), 2)

    def test_csv_with_header(self):
        with tempfile.TemporaryDirectory() as root:
            touch(root, "a.png")
            touch(root, "b.png")
            write_text(root, "labels.csv", "filename,text\na.png,%s\nb.png,\"%s, hai\"\n" % (TEXT_A, TEXT_B))
            records, _ = build_index(root)
        by_id = {r["id"]: r for r in records}
        self.assertEqual(by_id["a"]["gt_text"], TEXT_A)
        self.assertEqual(by_id["b"]["gt_text"], TEXT_B + ", hai")

    def test_sidecar_txt_files(self):
        with tempfile.TemporaryDirectory() as root:
            touch(root, "a.png")
            touch(root, "b.png")
            write_text(root, "a.txt", TEXT_A + "\n")
            write_text(root, "b.txt", TEXT_B + "\n")
            records, _ = build_index(root)
        self.assertEqual(sorted(r["gt_text"] for r in records), sorted([TEXT_A, TEXT_B]))

    def test_whitespace_runs_are_collapsed_and_nfc_applied(self):
        with tempfile.TemporaryDirectory() as root:
            touch(root, "a.png")
            write_text(root, "labels.txt", "a.png\tb" + "a\u0323" + "n   \u0111\u1ecdc\n")
            records, _ = build_index(root)
        self.assertEqual(records[0]["gt_text"], "b\u1ea1n \u0111\u1ecdc")

    def test_writer_regex(self):
        with tempfile.TemporaryDirectory() as root:
            touch(root, "w01_0001.png")
            touch(root, "w02_0001.png")
            write_text(root, "labels.txt", "w01_0001.png\t%s\nw02_0001.png\t%s\n" % (TEXT_A, TEXT_B))
            records, report = build_index(root, writer_regex=r"w(\d+)_")
        self.assertEqual({r["writer_id"] for r in records}, {"01", "02"})
        self.assertEqual(report["writers"], 2)

    def test_partial_split_folders_drop_official_split(self):
        with tempfile.TemporaryDirectory() as root:
            touch(root, "train/a.png")
            touch(root, "other/b.png")
            write_text(root, "labels.txt", "a.png\t%s\nb.png\t%s\n" % (TEXT_A, TEXT_B))
            records, report = build_index(root)
        self.assertTrue(all("split" not in r for r in records))
        self.assertIn("split_note", report)

    def test_no_labels_raises_with_diagnostics(self):
        with tempfile.TemporaryDirectory() as root:
            touch(root, "a.png")
            touch(root, "b.png")
            with self.assertRaises(IndexBuildError) as ctx:
                build_index(root)
        self.assertIn("first image paths", str(ctx.exception))

    def test_no_images_raises(self):
        with tempfile.TemporaryDirectory() as root:
            write_text(root, "labels.txt", "a.png\tx\n")
            with self.assertRaises(IndexBuildError):
                build_index(root)

    def test_output_feeds_make_splits(self):
        from scripts.make_splits import make_splits
        with tempfile.TemporaryDirectory() as root:
            for k in range(30):
                touch(root, "lines/%03d.png" % k)
            write_text(root, "labels.txt", "".join("%03d.png\t%s\n" % (k, TEXT_A) for k in range(30)))
            records, _ = build_index(root)
        result, mode = make_splits(records)
        self.assertEqual(mode, "by_id")
        self.assertEqual(len(result), 30)

    def test_group_regex_and_counts(self):
        with tempfile.TemporaryDirectory() as root:
            for page in ("1", "2"):
                touch(root, "train_data/%s/1.jpg" % page)
                write_text(root, "train_data/%s/label.json" % page, json.dumps({"1.jpg": TEXT_A}, ensure_ascii=False))
            records, report = build_index(root, group_regex=r"train_data/(\d+)/")
        self.assertEqual({r["group_id"] for r in records}, {"1", "2"})
        self.assertEqual(report["groups"], 2)

    def test_partial_group_ids_are_dropped(self):
        with tempfile.TemporaryDirectory() as root:
            touch(root, "a/1.jpg")
            touch(root, "b/1.jpg")
            write_text(root, "a/label.json", json.dumps({"1.jpg": TEXT_A}, ensure_ascii=False))
            write_text(root, "b/label.json", json.dumps({"1.jpg": TEXT_B}, ensure_ascii=False))
            records, report = build_index(root, group_regex=r"^(a)/")
        self.assertTrue(all("group_id" not in r for r in records))
        self.assertIn("group_note", report)

    def test_unlabeled_images_are_explained(self):
        with tempfile.TemporaryDirectory() as root:
            for k in range(40):
                touch(root, "p/%d.jpg" % k)
            labels = {"%d.jpg" % k: TEXT_A for k in range(38)}
            labels["38.jpg"] = "   "
            write_text(root, "p/label.json", json.dumps(labels, ensure_ascii=False))
            records, report = build_index(root)
        self.assertEqual(report["labeled"], 38)
        self.assertEqual(report["no_label_entry"], 1)
        self.assertEqual(report["empty_label"], 1)
        self.assertEqual(sorted(report["unlabeled_examples"]), ["p/38.jpg", "p/39.jpg"])

    def test_real_layout_end_to_end(self):
        """Mirrors UIT_HWDB_line: <root>/<top>/{train_data,test_data}/<page>/<n>.jpg plus label.json per page."""
        from scripts.make_splits import make_splits
        with tempfile.TemporaryDirectory() as root:
            pages = [("train_data", str(n)) for n in range(1, 21)] + [("test_data", str(n)) for n in range(21, 23)]
            for folder, page in pages:
                labels = {}
                for line in range(1, 11):
                    touch(root, "TOP/%s/%s/%d.jpg" % (folder, page, line))
                    labels["%d.jpg" % line] = "%s %s %d" % (TEXT_A, page, line)
                write_text(root, "TOP/%s/%s/label.json" % (folder, page), json.dumps(labels, ensure_ascii=False))
            records, report = build_index(root, group_regex=r"(?:train_data|test_data)/(\d+)/")
        self.assertEqual(report["labeled"], 220)
        self.assertEqual(report["split_counts"], {"train": 200, "test": 20})
        result, mode = make_splits(records)
        self.assertEqual(mode, "official+carved_val_by_group")
        by_group = {}
        for rec in records:
            by_group.setdefault(rec["group_id"], set()).add(result[rec["id"]])
        self.assertTrue(all(len(v) == 1 for v in by_group.values()))
        val_pages = sorted((g for g, v in by_group.items() if v == {"val"}), key=int)
        # 200 train lines, target 22 for val, whole pages only: pages 20, 19 and 18 (30 lines)
        self.assertEqual(val_pages, ["18", "19", "20"])
        self.assertEqual(sum(1 for v in result.values() if v == "val"), 30)


if __name__ == "__main__":
    unittest.main()
