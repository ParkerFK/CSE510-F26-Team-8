#import the necessary libraries
from __future__ import annotations
import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path
import chess
import numpy as np

#define the columns to keep and the vector dimension
KEEP = ["PuzzleId", "FEN", "Moves", "Rating", "Themes", "GameUrl"]

#define the vector dimension
VECTOR_DIM = 12 * 64

#define the three subset sizes
SUBSET_SIZES = (100_000, 500_000, 1_000_000)

#define the source URL to get the Lichess file
SOURCE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"

#define the source name and client name
SOURCE_NAME = "lichess_db_puzzle.csv.zst"
client_name = "CSE510-puzzle-loader"

#function to encode the FEN as a vector of 0s and 1s as a length-768 vector
def fen_to_vector(fen: str) -> np.ndarray:
    
    #create a chess board from the FEN string
    board = chess.Board(fen)
    
    #create a vector of 0s and 1s as a length-768 vector
    vec = np.zeros(VECTOR_DIM, dtype=np.uint8)
    
    #iterate over the pieces on the board
    for square, piece in board.piece_map().items():
        #get the index of the piece
        piece_index = (0 if piece.color == chess.WHITE else 6) + piece.piece_type - 1
        #set the bit in the vector
        vec[piece_index * 64 + square] = 1
    #return the vector
    return vec


#function to run the encoder checks
def run_encoder_checks() -> None:
    #check the fixed piece and square order on known positions
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    vec = fen_to_vector(start)
    
    #check if the starting position has 32 ones
    if int(vec.sum()) != 32:
        raise AssertionError(f"starting position should have 32 ones, got {int(vec.sum())}")
    #check if the white pawns occupy indexes 8 through 15
    if int(vec[8:16].sum()) != 8:
        raise AssertionError("white pawns should occupy indexes 8 through 15")
    #check if the white rook is on a1
    if vec[3 * 64 + chess.A1] != 1:
        raise AssertionError("white rook should sit on a1")
    #check if the empty board is all zeros
    empty = fen_to_vector("8/8/8/8/8/8/8/8 w - - 0 1")
    if int(empty.sum()) != 0:
        raise AssertionError("empty board should be all zeros")
    
    #print the results of the checks
    print("encoder checks passed")
    print("  starting position: 32 ones, white pawns at indexes 8-15, white rook on a1")

#function to decompress the .zst file for python 3.14+
def _zst_stream_reader(raw):
    try:
        from compression import zstd as std_zstd

        return std_zstd.ZstdDecompressor().stream_reader(raw)
    except ImportError:
        pass

    import shutil
    import subprocess

    zstd_bin = shutil.which("zstd")
    if zstd_bin is not None:
        #keep the handle open for the life of the pipe; caller closes `raw`
        proc = subprocess.Popen(
            [zstd_bin, "-d", "-c"],
            stdin=raw,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if proc.stdout is None:
            raise RuntimeError("zstd produced no stdout")
        return proc.stdout

    try:
        import zstandard as zstd
    except ImportError as exc:
        raise ImportError(
            "Need a .zst decompressor. Use Python 3.14+, install the "
            "`zstd` CLI, or `pip install zstandard` (may need Microsoft "
            "C++ Build Tools on Windows if no wheel is available)."
        ) from exc

    return zstd.ZstdDecompressor().stream_reader(raw)


#function to open a .csv or .csv.zst file as a text stream
def _open_text(src: Path):
    #check if the file is a .zst file
    if src.suffix == ".zst":
        #open the file as a binary stream
        raw = open(src, "rb")

        #create a zstandard reader (stdlib, CLI, or pip package)
        reader = _zst_stream_reader(raw)

        #create a text stream
        text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
        return text, raw

    #if it is not a .zst file, open the file as a text stream
    text = open(src, newline="", encoding="utf-8")
    return text, None

#function to load the 100k, 500k, and 1M subsets in one command run
def load_subsets(src: Path, out_dir: Path, sizes: tuple[int, ...]) -> dict[int, int]:
    #keep the sizes in ascending order so the last one is the stopping point
    sizes = tuple(sorted(set(sizes)))
    if not sizes or sizes[0] < 1:
        raise ValueError("subset sizes must be at least 1")

    #create the parent directory if it does not exist
    out_dir.mkdir(parents=True, exist_ok=True)

    #open the file as a text stream
    text, raw = _open_text(src)
    files = []

    #try to read the file
    try:
        #create a csv dictionary reader
        rows = csv.DictReader(text)

        #check if the file has a header row
        if rows.fieldnames is None:
            raise ValueError(f"{src} has no header row")

        #check if the file is missing any of the columns to keep
        missing = [name for name in KEEP if name not in rows.fieldnames]
        if missing:
            raise ValueError(f"{src} is missing columns: {', '.join(missing)}")

        #open one csv for each subset size
        writers = []
        for n in sizes:
            handle = open(out_dir / f"puzzles_{n}.csv", "w", newline="", encoding="utf-8")
            files.append(handle)
            writer = csv.DictWriter(handle, fieldnames=KEEP)
            writer.writeheader()
            writers.append((n, writer))

        kept = 0
        limit = sizes[-1]

        #iterate over the rows in the file
        for row in rows:
            #set the item to the row values for the columns to keep
            item = {name: row[name] for name in KEEP}
            kept += 1
            
            #the smaller files are prefixes of the larger ones
            for n, writer in writers:
                if kept <= n:
                    writer.writerow(item)
            if kept % 10_000 == 0:
                print(f"  loaded {kept:,} rows", flush=True)
            if kept >= limit:
                break
    finally:
        #close the subset files
        for handle in files:
            handle.close()
        #close the text stream
        text.close()
        #close the raw stream if it is not None
        if raw is not None:
            raw.close()

    if kept == 0:
        raise ValueError(f"{src} has a header but no puzzle rows")
    return {n: min(kept, n) for n in sizes}

#function to encode the largest CSV and save a vector file for each subset size
def encode_subsets(src: Path, out_dir: Path, sizes: tuple[int, ...]) -> list[tuple[int, Path, Path]]:
    #keep the sizes in ascending order
    sizes = tuple(sorted(set(sizes)))
    largest = sizes[-1]

    #create a numpy array of zeros as a length-768 vector for the vectors
    vectors = np.zeros((largest, VECTOR_DIM), dtype=np.uint8)

    #create a numpy array of empty strings for the ids
    ids = np.empty(largest, dtype="U16")

    #open the source file as a text stream
    with open(src, newline="", encoding="utf-8") as handle:
        #create a csv dictionary reader
        rows = csv.DictReader(handle)
        #check if the file is missing any of the columns to keep
        missing = [name for name in ("PuzzleId", "FEN") if name not in (rows.fieldnames or [])]
        #if it is, raise an error
        if missing:
            raise ValueError(f"{src} is missing columns: {', '.join(missing)}")
        kept = 0
        
        #iterate over the rows in the file
        for row in rows:
            if kept >= largest:
                break
            vectors[kept] = fen_to_vector(row["FEN"])
            ids[kept] = row["PuzzleId"]
            kept += 1
            if kept % 10_000 == 0:
                print(f"  encoded {kept:,} positions", flush=True)

    #if there are no puzzles, raise an error
    if kept == 0:
        raise ValueError(f"{src} has a header but no puzzle rows")

    #save each prefix so the smaller subsets stay aligned with their csv files
    saved = []
    
    #iterate over the subset sizes and save the vectors and ids
    for n in sizes:
        count = min(kept, n)
        vector_path = out_dir / f"vectors_{n}.npy"
        id_path = out_dir / f"ids_{n}.npy"
        np.save(vector_path, vectors[:count])
        np.save(id_path, ids[:count])
        saved.append((count, vector_path, id_path))
    return saved

#function to print a status alert in the terminal
def alert(message: str) -> None:
    #print a status alert
    print("\n" + "=" * 60, flush=True)
    print("ALERT", flush=True)
    print(message, flush=True)
    print("=" * 60 + "\n", flush=True)


#function to download the source file
def download_source(url: str, dest: Path) -> None:
    #create the parent directory if it does not exist
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    #create a partial destination file
    partial = dest.with_name(dest.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": client_name})
    
    #attempt to download the source file
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            read = 0
            next_report = 0
            #iterate over the response
            while True:
                #read a chunk of the response
                chunk = response.read(1024 * 1024)
                
                #if the chunk is empty, break the loop
                if not chunk:
                    break
                #write the chunk to the partial file
                out.write(chunk)
                #increment the number of bytes read
                read += len(chunk)
                
                #if the number of bytes read is greater than or equal to the next report, print the progress
                if read >= next_report:
                    #if the total size is not None, print the progress
                    if total:
                        #print the progress
                        print(
                            f"  downloaded {read / 1_000_000:.1f} / {total / 1_000_000:.1f} MB",
                            flush=True,
                        )
                    else:
                        print(f"  downloaded {read / 1_000_000:.1f} MB", flush=True)
                    next_report = read + 20_000_000
        
        #replace the partial file with the destination file
        partial.replace(dest)
   
    #if an exception occurs, print a message and raise the exception
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    
    #print the size of the destination file
    print(f"saved {dest} ({dest.stat().st_size / 1_000_000:.1f} MB)", flush=True)


#function to ensure the source file is present
def ensure_source(dest: Path) -> Path:
    #use the local file when it is already downloaded
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    #download the source file
    download_source(SOURCE_URL, dest)
    #return the destination file
    return dest

#function to parse the arguments from the command line
def parse_args(argv: list[str]) -> argparse.Namespace:
    #create an argument parser
    parser = argparse.ArgumentParser(
        description="Load a Lichess puzzle subset and encode each FEN as a 768-bit vector."
    )

    #add the --src argument
    parser.add_argument(
        "--src",
        type=Path,
        help="optional local .csv.zst or .csv. When omitted, the script downloads the Lichess file into sources/",
    )

    #add the --out argument
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="directory for the subset CSV and the .npy files (default: data)",
    )

    #add the --check argument
    parser.add_argument(
        "--check",
        action="store_true",
        help="run the encoder checks and exit",
    )

    #parse the arguments
    return parser.parse_args(argv)

#main function to run the script
def main(argv: list[str] | None = None) -> int:
    #parse the arguments
    args = parse_args(sys.argv[1:] if argv is None else argv)
    
    #if the --check argument is passed, run the encoder checks and exit
    if args.check:
        run_encoder_checks()
        return 0

    #if the --src argument is not passed, try to ensure the source file is present
    if args.src is None:
        try:
            args.src = ensure_source(Path("sources") / SOURCE_NAME)
        except Exception as exc:
            alert(f"The Lichess source could not be downloaded.\n\n{exc}")
            return 1
    #if the --src argument is passed, check if the file is a file
    elif not args.src.is_file():
        #print a message if the source file is not downloaded
        alert(
            "The source file you passed is not downloaded yet.\n\n"
            f"{args.src}\n\n"
            "Run the script with no arguments to download the Lichess file, "
            "or pass the path of a file that is already on disk.",
        )
        return 2

    #run the encoder checks
    run_encoder_checks()

    #create the output directory if it does not exist
    args.out.mkdir(parents=True, exist_ok=True)

    #load the 100k, 500k, and 1M csv files from the same rows
    print("loading 100,000, 500,000, and 1,000,000 puzzles from", args.src)
    written = load_subsets(args.src, args.out, SUBSET_SIZES)
    for n, count in written.items():
        print(f"wrote {count:,} rows to {args.out / f'puzzles_{n}.csv'}")

    #encode the 1M file once, then save the 100k and 500k prefixes
    largest_csv = args.out / f"puzzles_{SUBSET_SIZES[-1]}.csv"
    print(f"encoding {largest_csv}")
    for count, vector_path, id_path in encode_subsets(largest_csv, args.out, SUBSET_SIZES):
        vectors = np.load(vector_path, mmap_mode="r")
        print(f"encoded {count:,} positions")
        print(f"  {vector_path}  shape={vectors.shape} dtype={vectors.dtype}")
        print(f"  {id_path}  aligned with vectors row by row")
    return 0


#main function to run the script
#run the following usage in this order:
#python -m pip install -r requirements.txt (if those packages are not already installed)
#python prepare_puzzles.py
#python prepare_puzzles.py --check (optional to check the encoder checks)
if __name__ == "__main__":
    raise SystemExit(main())
