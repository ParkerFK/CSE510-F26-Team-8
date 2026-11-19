"""
Multi-Parameter Search Experiment & Benchmarking Runner for Chess Puzzle Search (Version 4).

Executes parameter grid search experiments across Plain Search, Search-First (Post-Filtering),
and Filter-First (Pre-Filtering) workflows using FAISS binary indexes and RocksDB.
Captures detailed performance metrics (latency, candidate overfetch count, result count, 
Hamming distance statistics) and exports results to CSV and JSON formats for analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Import directly from puzzle_search
try:
    import puzzle_search as puzzle_search_mod
except ImportError as e:
    sys.stderr.write(f"Error importing puzzle_search: {e}\n")
    sys.stderr.write(
        "Ensure puzzle_search.py, prepare_puzzles.py, faiss_indexes.py, and puzzle_store.py are in Python path.\n"
    )
    sys.exit(1)

PuzzleStore = puzzle_search_mod.PuzzleStore
PuzzleIndexes = puzzle_search_mod.PuzzleIndexes
store_path = puzzle_search_mod.store_path
ensure_indexes = getattr(puzzle_search_mod, "ensure_indexes", None)
plain_search = puzzle_search_mod.plain_search
search_first_then_filter = puzzle_search_mod.search_first_then_filter
filter_first_then_search = puzzle_search_mod.filter_first_then_search


# Standard test FEN positions for benchmarking
TEST_FENS = {
    "start_pos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "sicilian": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "french_defense": "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "tactical_midgame": "r1bqk2r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5",
}

# Parameter Grid Defaults
DEFAULT_K_VALUES = [5, 10, 20, 50]
DEFAULT_RATING_RANGES = [
    (None, None, "Unconstrained"),
    (1400, 1600, "Narrow (1400-1600)"),
    (1200, 1800, "Medium (1200-1800)"),
    (1000, 2200, "Wide (1000-2200)"),
]
DEFAULT_THEME_SETS = [
    (None, "all", "No Theme"),
    (["mateIn2"], "all", "Single Theme (mateIn2)"),
    (["fork", "advantage"], "all", "Multi Theme ALL (fork+advantage)"),
    (["fork", "skewer"], "any", "Multi Theme ANY (fork|skewer)"),
    (["pin", "endgame", "short"], "all", "Strict Multi Theme (pin+endgame+short)"),
]
DEFAULT_INDEX_TYPES = ["flat", "ivf", "hnsw"]


def extract_dist_stats(results: list[dict]) -> tuple[float | None, int | None, int | None]:
    """Safely extracts mean, min, and max Hamming distances from search result items."""
    valid_dists = []
    for r in results:
        dist_val = r.get("hamming_distance")
        if dist_val is not None:
            try:
                valid_dists.append(int(dist_val))
            except (ValueError, TypeError):
                pass

    if not valid_dists:
        return None, None, None

    avg_d = round(float(np.mean(valid_dists)), 2)
    min_d = int(np.min(valid_dists))
    max_d = int(np.max(valid_dists))
    return avg_d, min_d, max_d


def safe_filter_first_then_search(
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
    Safely executes filter-first search, handling unconstrained queries (allowed_rows is None)
    without passing None to np.ascontiguousarray.
    """
    allowed_rows = store.rows_for_filter(
        min_rating=min_rating,
        max_rating=max_rating,
        themes=themes,
        match=match,
    )

    if allowed_rows is None:
        results = plain_search(
            indexes=indexes,
            store=store,
            fen_string=fen_string,
            k=k,
            index_name=index_name,
        )
        return results, f"{index_name}(unconstrained)"

    return filter_first_then_search(
        indexes=indexes,
        store=store,
        fen_string=fen_string,
        k=k,
        min_rating=min_rating,
        max_rating=max_rating,
        themes=themes,
        match=match,
        index_name=index_name,
    )


def run_experiment_grid(
    data_dir: Path,
    size: int,
    k_values: list[int],
    rating_ranges: list[tuple[int | None, int | None, str]],
    theme_sets: list[tuple[list[str] | None, str, str]],
    index_types: list[str],
    repetitions: int = 3,
) -> list[dict[str, Any]]:
    """
    Executes search experiments over all parameter combinations and measures metrics.
    """
    db_path = store_path(data_dir, size)
    if not db_path.exists():
        raise FileNotFoundError(
            f"RocksDB store not found at {db_path}. Build it first using: "
            f"python puzzle_store.py build --size {size}"
        )

    # Ensure/load indexes
    if ensure_indexes is not None:
        indexes = ensure_indexes(index_dir=data_dir, size=size, required_index="flat")
    else:
        indexes = PuzzleIndexes.load(index_dir=data_dir, n=size)

    store = PuzzleStore(db_path)
    records: list[dict[str, Any]] = []

    total_runs = (
        len(TEST_FENS)
        * len(index_types)
        * len(k_values)
        * (1 + len(rating_ranges) * len(theme_sets) * 2)  # Plain + SearchFirst + FilterFirst
    )
    print(f"Starting parameter search experiment suite ({total_runs:,} total configurations)...")
    run_idx = 0

    with store:
        for fen_name, fen_str in TEST_FENS.items():
            for idx_type in index_types:
                if idx_type not in indexes.indexes:
                    print(f"Skipping index type '{idx_type}' (not loaded).")
                    continue

                for k in k_values:
                    # 1. PLAIN SEARCH
                    run_idx += 1
                    latencies = []
                    res_items = []
                    for _ in range(repetitions):
                        t0 = time.perf_counter()
                        res_items = plain_search(
                            indexes=indexes,
                            store=store,
                            fen_string=fen_str,
                            k=k,
                            index_name=idx_type,
                        )
                        latencies.append((time.perf_counter() - t0) * 1000)

                    avg_d, min_d, max_d = extract_dist_stats(res_items)
                    records.append({
                        "experiment_id": run_idx,
                        "fen_name": fen_name,
                        "search_mode": "plain",
                        "index_type": idx_type,
                        "k": k,
                        "min_rating": None,
                        "max_rating": None,
                        "rating_label": "None",
                        "themes": "None",
                        "match_rule": "N/A",
                        "theme_label": "None",
                        "results_returned": len(res_items),
                        "candidates_fetched": k,
                        "method_used": idx_type,
                        "avg_latency_ms": round(float(np.mean(latencies)), 4),
                        "min_latency_ms": round(float(np.min(latencies)), 4),
                        "max_latency_ms": round(float(np.max(latencies)), 4),
                        "avg_distance": avg_d,
                        "min_distance": min_d,
                        "max_distance": max_d,
                    })

                    # 2. FILTERED SEARCHES (Search-First & Filter-First)
                    for min_r, max_r, r_label in rating_ranges:
                        for th_list, match_rule, th_label in theme_sets:
                            th_str = ",".join(th_list) if th_list else "None"

                            # A. SEARCH-FIRST (Post-Filtering)
                            run_idx += 1
                            sf_latencies = []
                            sf_results = []
                            fetched_cnt = 0
                            for _ in range(repetitions):
                                t0 = time.perf_counter()
                                sf_results, fetched_cnt = search_first_then_filter(
                                    indexes=indexes,
                                    store=store,
                                    fen_string=fen_str,
                                    k=k,
                                    min_rating=min_r,
                                    max_rating=max_r,
                                    themes=th_list,
                                    match=match_rule,
                                    index_name=idx_type,
                                )
                                sf_latencies.append((time.perf_counter() - t0) * 1000)

                            sf_avg_d, sf_min_d, sf_max_d = extract_dist_stats(sf_results)
                            records.append({
                                "experiment_id": run_idx,
                                "fen_name": fen_name,
                                "search_mode": "search-first",
                                "index_type": idx_type,
                                "k": k,
                                "min_rating": min_r,
                                "max_rating": max_r,
                                "rating_label": r_label,
                                "themes": th_str,
                                "match_rule": match_rule,
                                "theme_label": th_label,
                                "results_returned": len(sf_results),
                                "candidates_fetched": fetched_cnt,
                                "method_used": "overfetch",
                                "avg_latency_ms": round(float(np.mean(sf_latencies)), 4),
                                "min_latency_ms": round(float(np.min(sf_latencies)), 4),
                                "max_latency_ms": round(float(np.max(sf_latencies)), 4),
                                "avg_distance": sf_avg_d,
                                "min_distance": sf_min_d,
                                "max_distance": sf_max_d,
                            })

                            # B. FILTER-FIRST (Pre-Filtering)
                            run_idx += 1
                            ff_latencies = []
                            ff_results = []
                            method_used = ""
                            for _ in range(repetitions):
                                t0 = time.perf_counter()
                                ff_results, method_used = safe_filter_first_then_search(
                                    indexes=indexes,
                                    store=store,
                                    fen_string=fen_str,
                                    k=k,
                                    min_rating=min_r,
                                    max_rating=max_r,
                                    themes=th_list,
                                    match=match_rule,
                                    index_name=idx_type,
                                )
                                ff_latencies.append((time.perf_counter() - t0) * 1000)

                            ff_avg_d, ff_min_d, ff_max_d = extract_dist_stats(ff_results)
                            records.append({
                                "experiment_id": run_idx,
                                "fen_name": fen_name,
                                "search_mode": "filter-first",
                                "index_type": idx_type,
                                "k": k,
                                "min_rating": min_r,
                                "max_rating": max_r,
                                "rating_label": r_label,
                                "themes": th_str,
                                "match_rule": match_rule,
                                "theme_label": th_label,
                                "results_returned": len(ff_results),
                                "candidates_fetched": len(ff_results),
                                "method_used": method_used,
                                "avg_latency_ms": round(float(np.mean(ff_latencies)), 4),
                                "min_latency_ms": round(float(np.min(ff_latencies)), 4),
                                "max_latency_ms": round(float(np.max(ff_latencies)), 4),
                                "avg_distance": ff_avg_d,
                                "min_distance": ff_min_d,
                                "max_distance": ff_max_d,
                            })

                    if run_idx % 50 == 0:
                        print(f"  Processed {run_idx}/{total_runs} experiment runs...", flush=True)

    print(f"Experiment grid completed: {len(records):,} total data points captured.")
    return records


def export_experiment_data(
    records: list[dict[str, Any]],
    output_csv: Path,
    output_json: Path,
) -> None:
    """Exports captured experiment metrics into CSV and structured JSON summary files."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. Export CSV
    fieldnames = list(records[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # 2. Compute Summary Aggregations
    modes = sorted(list({r["search_mode"] for r in records}))
    indexes = sorted(list({r["index_type"] for r in records}))

    summary_stats: dict[str, Any] = {
        "total_experiments": len(records),
        "by_search_mode": {},
        "by_index_type": {},
    }

    for m in modes:
        m_recs = [r for r in records if r["search_mode"] == m]
        lats = [r["avg_latency_ms"] for r in m_recs]
        summary_stats["by_search_mode"][m] = {
            "count": len(m_recs),
            "mean_latency_ms": round(float(np.mean(lats)), 4),
            "p50_latency_ms": round(float(np.median(lats)), 4),
            "p95_latency_ms": round(float(np.percentile(lats, 95)), 4),
        }

    for idx_t in indexes:
        idx_recs = [r for r in records if r["index_type"] == idx_t]
        lats = [r["avg_latency_ms"] for r in idx_recs]
        summary_stats["by_index_type"][idx_t] = {
            "count": len(idx_recs),
            "mean_latency_ms": round(float(np.mean(lats)), 4),
            "p50_latency_ms": round(float(np.median(lats)), 4),
            "p95_latency_ms": round(float(np.percentile(lats, 95)), 4),
        }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"Exported detailed results to: '{output_csv}'")
    print(f"Exported statistical summary to: '{output_json}'")


def print_summary_report(output_json: Path) -> None:
    """Prints a clean tabular summary of experiment metrics."""
    with open(output_json, encoding="utf-8") as f:
        data = json.load(f)

    print("\n" + "=" * 65)
    print(f"EXPERIMENT SUMMARY REPORT (Total Runs: {data['total_experiments']:,})")
    print("=" * 65)

    print("\n--- Latency Breakdown by Search Mode ---")
    print(f"{'Search Mode':<15} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10}")
    print("-" * 55)
    for mode, metrics in data["by_search_mode"].items():
        print(
            f"{mode:<15} | {metrics['mean_latency_ms']:<10.3f} | "
            f"{metrics['p50_latency_ms']:<10.3f} | {metrics['p95_latency_ms']:<10.3f}"
        )

    print("\n--- Latency Breakdown by Index Type ---")
    print(f"{'Index Type':<15} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10}")
    print("-" * 55)
    for idx_type, metrics in data["by_index_type"].items():
        print(
            f"{idx_type:<15} | {metrics['mean_latency_ms']:<10.3f} | "
            f"{metrics['p50_latency_ms']:<10.3f} | {metrics['p95_latency_ms']:<10.3f}"
        )
    print("=" * 65 + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-parameter search experiments and capture analysis data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100000,
        help="Dataset size prefix (e.g. 100000, 500000, 1000000).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing vector files and RocksDB store.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/benchmark_results.csv"),
        help="Path for saving detailed experiment data CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/benchmark_summary.json"),
        help="Path for saving summary JSON report.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of search iterations per configuration for latency averaging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        records = run_experiment_grid(
            data_dir=args.data_dir,
            size=args.size,
            k_values=DEFAULT_K_VALUES,
            rating_ranges=DEFAULT_RATING_RANGES,
            theme_sets=DEFAULT_THEME_SETS,
            index_types=DEFAULT_INDEX_TYPES,
            repetitions=args.repetitions,
        )

        export_experiment_data(
            records=records,
            output_csv=args.output_csv,
            output_json=args.output_json,
        )

        print_summary_report(args.output_json)

    except KeyboardInterrupt:
        print("\nExperiment run cancelled by user.")
        return 130
    except Exception as err:
        sys.stderr.write(f"\nExecution error during benchmark suite: {err}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
