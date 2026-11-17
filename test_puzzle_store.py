import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from puzzle_store import PuzzleStore, REQUIRED, build_store, main


class PuzzleStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / "puzzles_4.csv"
        self.ids = self.root / "ids_4.npy"
        self.db = self.root / "rocksdb_4"
        self.records = [
            dict(zip(REQUIRED, [pid, "8/8/8/8/8/8/8/8 w - - 0 1", "a2a3",
                                str(rating), themes, "https://lichess.org/example"]))
            for pid, rating, themes in [
                ("A", 1799, "fork"), ("B", 1800, "fork mate"),
                ("C", 1899, "mate"), ("D", 2200, ""),
            ]
        ]
        self.write_csv(self.records)
        np.save(self.ids, np.array(["A", "B", "C", "D"]))
        build_store(self.csv, self.db, self.ids, verbose=False)

    def write_csv(self, records, columns=REQUIRED):
        with self.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)

    def assert_original_store(self):
        with PuzzleStore(self.db) as store:
            self.assertEqual(store.n_rows, 4)
            self.assertEqual(store.get("A")["rating"], 1799)
            self.assertEqual(store.get_by_row(3)["id"], "D")

    def test_record_round_trip(self):
        with PuzzleStore(self.db) as store:
            self.assertEqual(store.get("B"), {
                "row": 1, "id": "B", "fen": self.records[1]["FEN"],
                "moves": "a2a3", "rating": 1800, "themes": ["fork", "mate"],
                "url": "https://lichess.org/example",
            })
            self.assertEqual(store.puzzle_id_of(2), "C")
            self.assertEqual(store.row_of("C"), 2)
            self.assertEqual(store.get_by_row(1), store.get("B"))
            self.assertIsNone(store.get("missing"))
            self.assertIsNone(store.row_of("missing"))
            for row in (-1, 4):
                with self.assertRaises(IndexError):
                    store.get_by_row(row)

    def test_theme_index(self):
        with PuzzleStore(self.db) as store:
            self.assertEqual(store.themes(), {"fork": 2, "mate": 2})
            self.assertEqual(store.rows_for_theme("fork").tolist(), [0, 1])
            self.assertEqual(store.rows_for_theme("missing").tolist(), [])

    def test_rating_ranges(self):
        cases = [
            (None, None, [0, 1, 2, 3]), (1800, 1899, [1, 2]),
            (1799, 1800, [0, 1]), (1800, 1800, [1]),
            (1900, 2199, []), (None, 1800, [0, 1]),
            (1899, None, [2, 3]), (None, 1000, []),
            (3000, None, []), (3000, 4000, []), (0, 1000, []),
        ]
        with PuzzleStore(self.db) as store:
            for low, high, expected in cases:
                with self.subTest(low=low, high=high):
                    self.assertEqual(store.rows_for_rating(low, high).tolist(), expected)

    def test_reversed_range_is_invalid(self):
        with PuzzleStore(self.db) as store:
            with self.assertRaises(ValueError):
                store.rows_for_rating(2200, 1800)

    def test_combined_filters(self):
        with PuzzleStore(self.db) as store:
            self.assertIsNone(store.rows_for_filter())
            self.assertEqual(store.rows_for_filter(themes=["fork", "mate"]).tolist(), [1])
            self.assertEqual(store.rows_for_filter(
                themes=["fork", "mate"], match="any").tolist(), [0, 1, 2])
            self.assertEqual(store.rows_for_filter(1800, 2200, "fork").tolist(), [1])
            self.assertEqual(store.rows_for_filter(themes="missing").tolist(), [])
            self.assertEqual(store.rows_for_filter(min_rating=3000).tolist(), [])
            with self.assertRaises(ValueError):
                store.rows_for_filter(match="invalid")

    def test_existing_store_requires_rebuild(self):
        with self.assertRaises(FileExistsError):
            build_store(self.csv, self.db, self.ids, verbose=False)
        self.assert_original_store()

    def test_successful_rebuild(self):
        self.records[0]["Rating"] = "2000"
        self.write_csv(self.records)
        build_store(self.csv, self.db, self.ids, rebuild=True, verbose=False)
        with PuzzleStore(self.db) as store:
            self.assertEqual(store.get("A")["rating"], 2000)
            self.assertEqual(store.rows_for_rating(2000, 2000).tolist(), [0])
        self.assertFalse(list(self.root.glob(".rocksdb_4-*")))

    def test_failed_rebuild_keeps_existing_store(self):
        for ids in (["wrong", "B", "C", "D"], ["A", "B", "C", "D", "E"]):
            with self.subTest(ids=ids):
                np.save(self.ids, np.array(ids))
                with self.assertRaises(ValueError):
                    build_store(self.csv, self.db, self.ids, rebuild=True, verbose=False)
                self.assert_original_store()
                self.assertFalse(list(self.root.glob(".rocksdb_4-*")))

    def test_missing_ids_keeps_existing_store(self):
        with self.assertRaises(FileNotFoundError):
            build_store(self.csv, self.db, self.root / "missing.npy",
                        rebuild=True, verbose=False)
        self.assert_original_store()

    def test_bad_csv_keeps_existing_store(self):
        bad_rating = dict(self.records[0], Rating="invalid")
        cases = [[], [self.records[0], self.records[0]], [bad_rating]]
        for records in cases:
            with self.subTest(records=records):
                self.write_csv(records)
                with self.assertRaises(ValueError):
                    build_store(self.csv, self.db, rebuild=True, verbose=False)
                self.assert_original_store()
                self.assertFalse(list(self.root.glob(".rocksdb_4-*")))

    def test_missing_column_keeps_existing_store(self):
        self.write_csv([], columns=["PuzzleId"])
        with self.assertRaises(ValueError):
            build_store(self.csv, self.db, rebuild=True, verbose=False)
        self.assert_original_store()

    def test_failed_first_build_leaves_no_store(self):
        target = self.root / "failed"
        np.save(self.ids, np.array(["wrong"]))
        with self.assertRaises(ValueError):
            build_store(self.csv, target, self.ids, verbose=False)
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob(".failed-*")))

    def test_failed_replacement_restores_previous_store(self):
        rename = Path.rename

        def fail_publish(source, target):
            if source.name == "new":
                raise OSError("simulated replacement failure")
            return rename(source, target)

        with patch.object(Path, "rename", fail_publish):
            with self.assertRaisesRegex(OSError, "replacement failure"):
                build_store(self.csv, self.db, self.ids, rebuild=True, verbose=False)
        self.assert_original_store()
        self.assertFalse(list(self.root.glob(".rocksdb_4-*")))

    def test_failed_restore_retains_backup(self):
        rename = Path.rename

        def fail_publish_and_restore(source, target):
            if source.name in ("new", "previous"):
                raise OSError("simulated rename failure")
            return rename(source, target)

        with patch.object(Path, "rename", fail_publish_and_restore):
            with self.assertRaises(OSError):
                build_store(self.csv, self.db, self.ids, rebuild=True, verbose=False)
        backups = list(self.root.glob(".rocksdb_4-*/previous"))
        self.assertEqual(len(backups), 1)
        with PuzzleStore(backups[0]) as store:
            self.assertEqual(store.get("A")["rating"], 1799)

    def test_batch_boundaries(self):
        with patch("puzzle_store.BATCH_SIZE", 2):
            build_store(self.csv, self.db, self.ids, rebuild=True, verbose=False)
        self.assert_original_store()

    def test_invalid_bucket_width(self):
        for width in (0, -1):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    build_store(self.csv, self.db, bucket_width=width,
                                rebuild=True, verbose=False)
                self.assert_original_store()

    def test_cli(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["lookup", "--size", "4", "--data", str(self.root),
                                   "--id", "B"]), 0)
        self.assertEqual(json.loads(output.getvalue())["id"], "B")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["filter", "--size", "4", "--data", str(self.root),
                                   "--min-rating", "3000"]), 0)
        self.assertIn("0 of 4 puzzles match", output.getvalue())


if __name__ == "__main__":
    unittest.main()
