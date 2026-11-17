# CSE510-F26-Team-8

Install dependencies and prepare the puzzle subsets:

```sh
python -m pip install -r requirements.txt
python prepare_puzzles.py
```

Build a RocksDB store for a prepared subset:

```sh
python puzzle_store.py build --size 100000
```

Use `--size 500000` or `--size 1000000` for the other subsets, and `--data PATH`
for a different data directory. Add `--rebuild` to replace an existing store.
A failed build keeps the previous store.

Look up a puzzle or filter by rating and theme:

```sh
python puzzle_store.py lookup --size 100000 --id 00sHx
python puzzle_store.py filter --size 100000 --min-rating 1800 --max-rating 2200 --theme fork
```

Search code can import `PuzzleStore` and use `get_by_row()` to retrieve a full
record, or `rows_for_filter()` to get eligible rows. Rating bounds are inclusive.
Multiple themes match all by default; use `match="any"` (CLI: `--match any`) for
an OR filter. No filter returns `None`; no matches returns an empty array.

Store rows match the original subset CSV and `vectors_N.npy` / `ids_N.npy` order.
Faiss indexes must preserve those IDs. Theme and rating indexes contain row IDs;
use `puzzle_id_of()` to resolve them to PuzzleIds. A temporary Faiss index built
from filtered rows needs a mapping back to the original rows.

Run the storage tests and encoder checks without downloading the dataset:

```sh
python -m unittest discover -p 'test_*.py' -v
python prepare_puzzles.py --check
```
