"""
Binary FAISS Search Workflow Script for Chess Puzzles (Version 3).

Combines Plain Search, Search-First (Post-Filtering with Overfetching),
and Filter-First (Pre-Filtering with FAISS IDSelectorBatch) into a single
unified Python script with CLI argument parsing, automatic index detection
and building, and robust error handling.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Import core utilities from project modules
try:
    from prepare_puzzles import fen_to_vector
    from faiss_indexes import (
        pack_vector,
        PuzzleIndexes,
        load_packed_vectors,
        load_ids,
    )
    from puzzle_store import PuzzleStore, store_path
except ImportError as e:
    sys.stderr.write(f"Error importing dependencies: {e}\n")
    sys.stderr.write(
        "Ensure prepare_puzzles.py, faiss_indexes.py, and puzzle_store.py are in Python path.\n"
    )


def encode_position(fen_string: str) -> np.ndarray:
    """
    Encodes a FEN chess position string into a 96-byte packed binary vector (768 bits).

    Args:
        fen_string: FEN string representing chess board state.

    Returns:
        Packed uint8 numpy array of shape (1, 96).
    """
    try:
        bit_vector = fen_to_vector(fen_string)
        packed_query = pack_vector(bit_vector)
        return packed_query
    except Exception as err:
        raise ValueError(f"Failed to encode FEN string '{fen_string}': {err}") from err


def ensure_indexes(
    index_dir: Path, size: int, required_index: str = "flat"
) -> PuzzleIndexes:
    """
    Checks if FAISS index files and ID mapping exist for the given size.
    If missing, automatically builds them from vectors_{size}.npy and ids_{size}.npy
    and saves them to disk.

    Args:
        index_dir: Path to directory containing vector/index files.
        size: Dataset size prefix (e.g. 100000).
        required_index: Specific index structure required ('flat', 'ivf', or 'hnsw').

    Returns:
        Loaded PuzzleIndexes instance.
    """
    index_dir = Path(index_dir)
    idmap_path = index_dir / f"idmap_{size}.npy"
    req_index_path = index_dir / f"{required_index}_{size}.index"

    need_build = False
    if not idmap_path.exists() or not req_index_path.exists():
        need_build = True
    else:
        try:
            indexes = PuzzleIndexes.load(index_dir=index_dir, n=size)
            if required_index not in indexes.indexes:
                need_build = True
            else:
                return indexes
        except Exception:
            need_build = True

    if need_build:
        print(
            f"Index files or idmap for size={size:,} not found in '{index_dir}'. "
            f"Automatically building FAISS indexes..."
        )
        vec_path = index_dir / f"vectors_{size}.npy"
        id_path = index_dir / f"ids_{size}.npy"

        if not vec_path.is_file() or not id_path.is_file():
            raise FileNotFoundError(
                f"Cannot build indexes automatically: vector or ID files "
                f"('vectors_{size}.npy', 'ids_{size}.npy') are missing in '{index_dir}'. "
                f"Please run 'python prepare_puzzles.py' first to generate vector datasets."
            )

        print(f"  Loading packed vectors from '{vec_path}'...")
        packed = load_packed_vectors(vec_path)

        print(f"  Loading puzzle IDs from '{id_path}'...")
        ids = load_ids(id_path)

        print(f"  Building FAISS indexes (Flat, IVF, HNSW) for {len(ids):,} puzzles...")
        t0 = time.perf_counter()
        indexes = PuzzleIndexes.build(packed=packed, ids=ids)
        elapsed = time.perf_counter() - t0
        print(f"  Indexes built successfully in {elapsed:.2f} seconds.")

        print(f"  Saving index files and 'idmap_{size}.npy' to '{index_dir}'...")
        indexes.save(index_dir, n=size)
        print("  Index build complete and saved to disk.\n")
        return indexes

    return PuzzleIndexes.load(index_dir=index_dir, n=size)


def plain_search(
    indexes: PuzzleIndexes,
    store: PuzzleStore,
    fen_string: str,
    k: int = 5,
    index_name: str = "flat",
) -> list[dict]:
    """
    Task 1: Plain Search.
    Encodes the FEN position, searches the FAISS binary index for the top-k nearest neighbors,
    and fetches full puzzle records from RocksDB.
    """
    query_vec = encode_position(fen_string)
    puzzle_ids, distances = indexes.search(name=index_name, query=query_vec, k=k)

    results = []
    for pid, dist in zip(puzzle_ids, distances):
        record = store.get(pid)
        if record is not None:
            record["hamming_distance"] = dist
            results.append(record)

    return results


def search_first_then_filter(
    indexes: PuzzleIndexes,
    store: PuzzleStore,
    fen_string: str,
    k: int = 5,
    min_rating: int | None = None,
    max_rating: int | None = None,
    themes: list[str] | None = None,
    match: str = "all",
    index_name: str = "flat",
    start_factor: int = 4,
) -> tuple[list[dict], int]:
    """
    Task 2A: Filtered Search (Search First, Then Filter / Post-Filtering).
    Overfetches candidate positions from FAISS iteratively, filtering candidates via keep_fn predicate
    until k matching puzzles are found.
    """
    query_vec = encode_position(fen_string)

    def keep_fn(puzzle_id: str) -> bool:
        record = store.get(puzzle_id)
        if record is None:
            return False

        # Rating filtering
        rating = record.get("rating")
        if min_rating is not None and rating < min_rating:
            return False
        if max_rating is not None and rating > max_rating:
            return False

        # Theme filtering
        if themes:
            rec_themes = set(record.get("themes", []))
            if match == "all":
                if not all(t in rec_themes for t in themes):
                    return False
            elif match == "any":
                if not any(t in rec_themes for t in themes):
                    return False

        return True

    matched_ids, distances, fetched_count = indexes.search_overfetch(
        name=index_name,
        query=query_vec,
        k=k,
        keep_fn=keep_fn,
        start_factor=start_factor,
    )

    results = []
    for pid, dist in zip(matched_ids, distances):
        record = store.get(pid)
        if record is not None:
            record["hamming_distance"] = dist
            results.append(record)

    return results, fetched_count


def filter_first_then_search(
    indexes: PuzzleIndexes,
    store: PuzzleStore,
    fen_string: str,
    k: int = 5,
    min_rating: int | None = None,
    max_rating: int | None = None,
    themes: list[str] | None = None,
    match: str = "all",
    index_name: str = "flat",
) -> tuple[list[dict], str]:
    """
    Task 2B: Filtered Search (Filter First, Then Search / Pre-Filtering).
    Queries RocksDB inverted index for matching candidate row indices, then restricts FAISS binary
    search to those candidate rows using faiss.IDSelectorBatch.
    """
    allowed_rows = store.rows_for_filter(
        min_rating=min_rating,
        max_rating=max_rating,
        themes=themes,
        match=match,
    )

    # If no filter is specified, store.rows_for_filter returns None (unconstrained).
    # Perform standard plain search to avoid passing None to np.ascontiguousarray in FAISS.
    if allowed_rows is None:
        results = plain_search(
            indexes=indexes,
            store=store,
            fen_string=fen_string,
            k=k,
            index_name=index_name,
        )
        return results, f"{index_name}(unconstrained)"

    if len(allowed_rows) == 0:
        return [], "empty_filter_match"

    query_vec = encode_position(fen_string)

    matched_ids, distances, method_used = indexes.search_subset(
        name=index_name,
        query=query_vec,
        k=k,
        allowed_faiss_ids=allowed_rows,
    )

    results = []
    for pid, dist in zip(matched_ids, distances):
        record = store.get(pid)
        if record is not None:
            record["hamming_distance"] = dist
            results.append(record)

    return results, method_used


def print_results(results: list[dict], title: str, extra_info: str = "") -> None:
    """Helper to cleanly display search results in CLI."""
    header = f"=== {title} ==="
    if extra_info:
        header += f" ({extra_info})"
    print(header)
    if not results:
        print("  No matching puzzles found.\n")
        return

    for idx, rec in enumerate(results, start=1):
        pid = rec.get("id", rec.get("PuzzleId", "N/A"))
        dist = rec.get("hamming_distance", "N/A")
        rating = rec.get("rating", rec.get("Rating", "N/A"))
        rec_themes = " ".join(rec.get("themes", []))
        fen = rec.get("fen", rec.get("FEN", "N/A"))
        print(f"  {idx}. ID: {pid:<10} Hamming Dist: {dist:<4} Rating: {rating:<5} Themes: {rec_themes}")
        print(f"     FEN: {fen}")
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search binary FAISS index and RocksDB puzzle store using Plain and Filtered workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--fen",
        type=str,
        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        help="FEN chess position query string.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbor puzzles to retrieve.",
    )
    parser.add_argument(
        "--mode",
        choices=["plain", "search-first", "filter-first", "all"],
        default="all",
        help="Search mode to execute.",
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf", "hnsw"],
        default="flat",
        help="FAISS index structure to query.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100000,
        help="Dataset size prefix (e.g., 100000, 500000, 1000000).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing FAISS indexes and RocksDB stores.",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=None,
        help="Minimum puzzle rating threshold.",
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=None,
        help="Maximum puzzle rating threshold.",
    )
    parser.add_argument(
        "--theme",
        action="append",
        dest="themes",
        help="Theme filter (can be specified multiple times).",
    )
    parser.add_argument(
        "--match",
        choices=["all", "any"],
        default="all",
        help="Theme match rule when multiple themes are specified.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    db_path = store_path(args.data_dir, args.size)

    # 1. Validation & Resource Loading
    if not db_path.exists():
        sys.stderr.write(f"Error: RocksDB store not found at '{db_path}'.\n")
        sys.stderr.write(
            f"Please build the store first using: python puzzle_store.py build --size {args.size}\n"
        )
        return 1

    try:
        # Load indexes or build them automatically if missing
        indexes = ensure_indexes(
            index_dir=args.data_dir, size=args.size, required_index=args.index_type
        )
    except Exception as err:
        sys.stderr.write(f"Error ensuring/loading FAISS indexes: {err}\n")
        return 1

    try:
        print(f"Opening RocksDB store from '{db_path}'...")
        store = PuzzleStore(db_path)
    except Exception as err:
        sys.stderr.write(f"Error opening RocksDB store: {err}\n")
        return 1

    # 2. Execution Routing
    try:
        with store:
            print(f"\nQuery FEN: {args.fen}")
            print(f"Index: {args.index_type} | Top-K: {args.k}\n")

            # --- PLAIN SEARCH ---
            if args.mode in ("plain", "all"):
                t0 = time.perf_counter()
                plain_res = plain_search(
                    indexes=indexes,
                    store=store,
                    fen_string=args.fen,
                    k=args.k,
                    index_name=args.index_type,
                )
                elapsed = (time.perf_counter() - t0) * 1000
                print_results(plain_res, "Plain Search", f"Time: {elapsed:.2f} ms")

            # --- SEARCH FIRST, THEN FILTER ---
            if args.mode in ("search-first", "all"):
                t0 = time.perf_counter()
                sf_res, fetched_cnt = search_first_then_filter(
                    indexes=indexes,
                    store=store,
                    fen_string=args.fen,
                    k=args.k,
                    min_rating=args.min_rating,
                    max_rating=args.max_rating,
                    themes=args.themes,
                    match=args.match,
                    index_name=args.index_type,
                )
                elapsed = (time.perf_counter() - t0) * 1000
                info_str = f"Fetched candidates: {fetched_cnt}, Time: {elapsed:.2f} ms"
                print_results(
                    sf_res, "Search First, Then Filter (Post-Filtering)", info_str
                )

            # --- FILTER FIRST, THEN SEARCH ---
            if args.mode in ("filter-first", "all"):
                t0 = time.perf_counter()
                ff_res, method = filter_first_then_search(
                    indexes=indexes,
                    store=store,
                    fen_string=args.fen,
                    k=args.k,
                    min_rating=args.min_rating,
                    max_rating=args.max_rating,
                    themes=args.themes,
                    match=args.match,
                    index_name=args.index_type,
                )
                elapsed = (time.perf_counter() - t0) * 1000
                info_str = f"Execution method: {method}, Time: {elapsed:.2f} ms"
                print_results(
                    ff_res, "Filter First, Then Search (Pre-Filtering)", info_str
                )

    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
        return 130
    except Exception as err:
        sys.stderr.write(f"\nExecution error: {err}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
