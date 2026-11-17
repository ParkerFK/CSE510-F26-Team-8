"""Puzzle records and filter indexes backed by RocksDB.

Rows match the original CSV and vectors/ids .npy files. Faiss indexes must
preserve those row IDs. Theme and rating indexes store rows, not PuzzleIds.

Keys: p:<PuzzleId>, r:<row>, theme:<name>, rb:<bucket>, meta:ratings, meta:info.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rocksdict import AccessType, Options, Rdict, WriteBatch

REQUIRED = ["PuzzleId", "FEN", "Moves", "Rating", "Themes", "GameUrl"]

BUCKET_WIDTH = 100

BATCH_SIZE = 10_000


def store_path(data_dir: Path, size: int) -> Path:
    return Path(data_dir) / f"rocksdb_{size}"


def rows_to_bytes(rows: list[int]) -> bytes:
    return np.array(sorted(rows), dtype=np.int32).tobytes()


def bytes_to_rows(raw: bytes | None) -> np.ndarray:
    if raw is None:
        return np.empty(0, dtype=np.int32)
    return np.frombuffer(raw, dtype=np.int32)


def build_store(csv_path: Path, db_path: Path, ids_path: Path | None = None,
                bucket_width: int = BUCKET_WIDTH, rebuild: bool = False, verbose: bool = True) -> dict:
    csv_path, db_path = Path(csv_path), Path(db_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"subset csv not found: {csv_path}")
    if bucket_width < 1:
        raise ValueError("bucket_width must be positive")
    if db_path.exists() and not rebuild:
        raise FileExistsError(f"{db_path} already exists, use --rebuild to replace it")

    expected_ids = np.load(ids_path) if ids_path is not None else None
    db_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f".{db_path.name}-", dir=db_path.parent))
    staged = work_dir / "new"
    previous = work_dir / "previous"
    published = False

    try:
        info = _write_store(csv_path, staged, expected_ids, bucket_width, verbose)
        if db_path.exists():
            if not rebuild:
                raise FileExistsError(f"{db_path} already exists, use --rebuild to replace it")
            db_path.rename(previous)
        try:
            staged.rename(db_path)
        except OSError:
            if previous.exists():
                previous.rename(db_path)
            raise
        published = True
    finally:
        # Keep the old database if restoring it failed.
        if published or not previous.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    if verbose:
        print(f"built {db_path} with {info['n_rows']:,} puzzles in {info['build_seconds']}s")
    return info


def _write_store(csv_path: Path, db_path: Path, expected_ids: np.ndarray | None,
                 bucket_width: int, verbose: bool) -> dict:
    options = Options()
    options.create_if_missing(True)
    db = Rdict(str(db_path), options)

    start = time.perf_counter()
    theme_rows = defaultdict(list)
    bucket_rows = defaultdict(list)
    ratings = []
    seen = set()

    try:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in REQUIRED if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

            batch = WriteBatch()
            row = -1
            for row, rec in enumerate(reader):
                pid = rec["PuzzleId"]

                if pid in seen:
                    raise ValueError(f"duplicate PuzzleId {pid} at row {row}")
                seen.add(pid)

                if expected_ids is not None:
                    if row >= len(expected_ids) or str(expected_ids[row]) != pid:
                        raise ValueError(f"row {row} of the csv does not match the ids file")

                try:
                    rating = int(rec["Rating"])
                except ValueError:
                    raise ValueError(f"row {row} has a bad Rating: {rec['Rating']!r}")

                themes = rec["Themes"].split()
                record = {"row": row, "id": pid, "fen": rec["FEN"], "moves": rec["Moves"],
                          "rating": rating, "themes": themes, "url": rec["GameUrl"]}
                batch.put(f"p:{pid}", json.dumps(record))
                batch.put(f"r:{row}", pid)

                ratings.append(rating)
                bucket_rows[rating // bucket_width].append(row)
                for theme in themes:
                    theme_rows[theme].append(row)

                if (row + 1) % BATCH_SIZE == 0:
                    db.write(batch)
                    batch = WriteBatch()
                    if verbose:
                        print(f"  stored {row + 1:,} puzzles", flush=True)
            db.write(batch)
            n_rows = row + 1

        if n_rows == 0:
            raise ValueError(f"{csv_path} has a header but no puzzle rows")
        if expected_ids is not None and len(expected_ids) != n_rows:
            raise ValueError(f"the ids file has {len(expected_ids)} ids but the csv has {n_rows} rows")

        batch = WriteBatch()
        for theme, rows in theme_rows.items():
            batch.put(f"theme:{theme}", rows_to_bytes(rows))
        for bucket, rows in bucket_rows.items():
            batch.put(f"rb:{bucket}", rows_to_bytes(rows))
        batch.put("meta:ratings", np.array(ratings, dtype=np.int16).tobytes())

        info = {"n_rows": n_rows, "bucket_width": bucket_width,
                "min_rating": min(ratings), "max_rating": max(ratings),
                "themes": {t: len(r) for t, r in sorted(theme_rows.items())},
                "build_seconds": round(time.perf_counter() - start, 3)}
        batch.put("meta:info", json.dumps(info))
        db.write(batch)
        db.flush()
    finally:
        db.close()

    return info


class PuzzleStore:
    def __init__(self, db_path: Path | str):
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"no store at {db_path}")
        self.path = db_path
        self.db = Rdict(str(db_path), access_type=AccessType.read_only())
        self.info = json.loads(self.db["meta:info"])
        self.n_rows = self.info["n_rows"]
        self.bucket_width = self.info["bucket_width"]
        self._ratings = None

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "PuzzleStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def get(self, puzzle_id: str) -> dict | None:
        raw = self.db.get(f"p:{puzzle_id}")
        return None if raw is None else json.loads(raw)

    def puzzle_id_of(self, row: int) -> str:
        row = int(row)
        if not 0 <= row < self.n_rows:
            raise IndexError(f"row {row} is out of range, the store has {self.n_rows} rows")
        return self.db[f"r:{row}"]

    def row_of(self, puzzle_id: str) -> int | None:
        rec = self.get(puzzle_id)
        return None if rec is None else rec["row"]

    def get_by_row(self, row: int) -> dict:
        return self.get(self.puzzle_id_of(row))

    @property
    def ratings(self) -> np.ndarray:
        if self._ratings is None:
            self._ratings = np.frombuffer(self.db["meta:ratings"], dtype=np.int16)
        return self._ratings

    def themes(self) -> dict[str, int]:
        return dict(self.info["themes"])

    def rows_for_theme(self, theme: str) -> np.ndarray:
        return bytes_to_rows(self.db.get(f"theme:{theme}"))

    def rows_for_rating(self, min_rating: int | None = None, max_rating: int | None = None) -> np.ndarray:
        """Return rows within the inclusive rating range."""
        low = None if min_rating is None else int(min_rating)
        high = None if max_rating is None else int(max_rating)
        if low is not None and high is not None and low > high:
            raise ValueError(f"min_rating {low} is bigger than max_rating {high}")

        low = self.info["min_rating"] if low is None else max(low, self.info["min_rating"])
        high = self.info["max_rating"] if high is None else min(high, self.info["max_rating"])
        if low > high:
            return np.empty(0, dtype=np.int32)

        parts = [bytes_to_rows(self.db.get(f"rb:{b}"))
                 for b in range(low // self.bucket_width, high // self.bucket_width + 1)]
        rows = np.concatenate(parts)
        keep = (self.ratings[rows] >= low) & (self.ratings[rows] <= high)
        return np.sort(rows[keep])

    def rows_for_filter(self, min_rating: int | None = None, max_rating: int | None = None,
                        themes: list[str] | str | None = None, match: str = "all") -> np.ndarray | None:
        """Return matching rows, or None when no filter is supplied.

        match="all" requires every theme; match="any" accepts at least one.
        """
        if match not in ("all", "any"):
            raise ValueError("match has to be 'all' or 'any'")
        if isinstance(themes, str):
            themes = [themes]
        use_rating = min_rating is not None or max_rating is not None
        if not use_rating and not themes:
            return None

        rows = None
        if themes:
            lists = [self.rows_for_theme(t) for t in themes]
            if match == "all":
                rows = lists[0]
                for other in lists[1:]:
                    rows = np.intersect1d(rows, other, assume_unique=True)
            else:
                rows = np.unique(np.concatenate(lists))
        if use_rating:
            by_rating = self.rows_for_rating(min_rating, max_rating)
            rows = by_rating if rows is None else np.intersect1d(rows, by_rating, assume_unique=True)
        return rows


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query the RocksDB puzzle store.")
    parser.add_argument("command", choices=["build", "lookup", "filter"])
    parser.add_argument("--size", type=int, required=True, help="subset size, for example 100000")
    parser.add_argument("--data", type=Path, default=Path("data"), help="data folder (default: data)")
    parser.add_argument("--rebuild", action="store_true", help="replace a store that already exists")
    parser.add_argument("--id", help="PuzzleId for the lookup command")
    parser.add_argument("--min-rating", type=int)
    parser.add_argument("--max-rating", type=int)
    parser.add_argument("--theme", action="append", help="can be used more than once")
    parser.add_argument("--match", choices=["all", "any"], default="all")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    db_path = store_path(args.data, args.size)

    if args.command == "build":
        csv_path = args.data / f"puzzles_{args.size}.csv"
        ids_path = args.data / f"ids_{args.size}.npy"
        if not csv_path.is_file():
            print(f"{csv_path} is missing, run prepare_puzzles.py first")
            return 2
        build_store(csv_path, db_path, ids_path if ids_path.is_file() else None, rebuild=args.rebuild)
        return 0

    if not db_path.exists():
        print(f"no store at {db_path}, run: python puzzle_store.py build --size {args.size}")
        return 2
    with PuzzleStore(db_path) as store:
        if args.command == "lookup":
            if not args.id:
                print("lookup needs --id")
                return 2
            print(json.dumps(store.get(args.id), indent=2))
        else:
            rows = store.rows_for_filter(args.min_rating, args.max_rating, args.theme, args.match)
            count = store.n_rows if rows is None else len(rows)
            print(f"{count:,} of {store.n_rows:,} puzzles match")
            for row in ([] if rows is None else rows[:5]):
                rec = store.get_by_row(row)
                print(f"  {rec['id']}  rating={rec['rating']}  themes={' '.join(rec['themes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
