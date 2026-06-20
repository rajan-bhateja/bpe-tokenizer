import os
import json
import regex
from collections import Counter
from logger import get_logger


LOGGER = get_logger(__name__)


# CONSTANTS
TRAINING_DATA_DIR = "books"
VOCAB_SIZE = 3_500
MODEL_OUTPUT_PATH = "tokenizer.json"
GPT2_REGEX_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def load_books(data_dir: str) -> tuple[list[str], int]:
    """Returns a list of .txt book filenames in the specified directory, along with the total count."""
    try:
        if not os.path.exists(data_dir):
            LOGGER.error(f"'{data_dir}' directory does not exist. Please check the path and try again.")
            return [], 0

        LOGGER.info(f"'{data_dir}' directory exists. Loading training data...")
        books_list = [book for book in os.listdir(data_dir) if book.endswith(".txt")]
        n_books = len(books_list)

        if n_books == 0:
            LOGGER.warning(f"Directory '{data_dir}' appears to be empty.")
            return [], 0

        return books_list, n_books

    except Exception as e:
        LOGGER.error(f"An error occurred while loading books: {e}")
        return [], 0


def read_books(books_list: list[str], data_dir: str) -> str:
    """Reads the content of each book in the list and returns a single concatenated corpus string."""
    try:
        corpus_parts = []
        for book in books_list:
            book_path = os.path.join(data_dir, book)
            LOGGER.info(f"Reading book: {book_path}")

            with open(book_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content:
                LOGGER.warning(f"Book '{book}' is empty. Skipping...")
                continue

            corpus_parts.append(content)
            LOGGER.info(f"Successfully read '{book}' ({len(content)} characters).")

        return "".join(corpus_parts)

    except Exception as e:
        LOGGER.error(f"An error occurred while reading books: {e}")
        return ""


def build_frequency_table(corpus: str) -> dict[tuple[int, ...], int]:
    """Builds a frequency table of byte sequences from the corpus using the GPT-2 regex pattern."""
    chunks = regex.findall(GPT2_REGEX_PATTERN, corpus)
    chunks_count = Counter(chunks)

    frequency_table = {
        tuple(chunk.encode("utf-8")): count
        for chunk, count in chunks_count.items()
    }

    LOGGER.info(f"Total chunks processed: {sum(chunks_count.values())}")
    LOGGER.info(f"Unique byte sequences: {len(frequency_table)}")
    return frequency_table


def get_pair_counts(frequency_table: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    """Counts the frequency of every adjacent pair of token IDs across all sequences."""
    pair_counts: dict[tuple[int, int], int] = {}

    for sequence, freq in frequency_table.items():
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq

    return pair_counts


def merge_sequence(
        sequence: tuple[int, ...],
        pair: tuple[int, int],
        new_id: int
    ) -> tuple[int, ...]:
    """Replaces all occurrences of the given adjacent pair in a sequence with new_id."""
    result = []
    i = 0
    while i < len(sequence):
        if i < len(sequence) - 1 and sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(sequence[i])
            i += 1

    return tuple(result)


def train(
        frequency_table: dict[tuple[int, ...], int],
        vocab_size: int
    ) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]:
    """Runs the BPE merge loop and returns the merge table and final vocabulary."""
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: dict[tuple[int, int], int] = {}
    next_id = 256
    n_merges = vocab_size - 256

    LOGGER.info(f"Starting BPE training: {n_merges} merges to perform.")

    for i in range(n_merges):
        pair_counts = get_pair_counts(frequency_table)

        if not pair_counts:
            LOGGER.warning(f"No more pairs to merge after {i} iterations. Stopping early.")
            break

        # Find the most frequent pair
        top_pair_item = max(pair_counts.items(), key=lambda item: item[1])
        top_pair = top_pair_item[0]
        top_count = top_pair_item[1]

        if top_count == 0:
            LOGGER.warning(f"Top pair count is 0 after {i} iterations. Stopping early.")
            break

        new_id = next_id
        next_id += 1
        merges[top_pair] = new_id
        vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]

        frequency_table = {
            merge_sequence(sequence, top_pair, new_id): freq
            for sequence, freq in frequency_table.items()
        }

        if (i + 1) % 100 == 0:
            LOGGER.info(
                f"Merge {i + 1}/{n_merges} | "
                f"Pair: {top_pair} -> {new_id} | "
                f"Token: {vocab[new_id]} | "
                f"Count: {top_count}"
            )

    LOGGER.info(f"Training complete. Vocabulary size: {len(vocab)} | Merges recorded: {len(merges)}")
    return merges, vocab


def save_tokenizer(
        merges: dict[tuple[int, int], int],
        vocab: dict[int, bytes],
        output_path: str
    ) -> None:
    """Serializes the vocabulary to a token-centric JSON file."""
    try:
        # Build a reverse lookup: token_id -> (id_a, id_b) that produced it
        # Base byte tokens (0-255) have no merges
        merge_parents: dict[int, tuple[int, int]] = {
            new_id: pair
            for pair, new_id in merges.items()
        }

        # Build a rank lookup: token_id -> merge_rank (0-indexed order it was learned)
        # Base byte tokens have no rank
        merge_rank: dict[int, int] = {
            new_id: rank
            for rank, (_, new_id) in enumerate(merges.items())
        }

        tokens = {}
        for token_id, byte_seq in vocab.items():
            entry: dict = {
                "content": byte_seq.decode("utf-8", errors="replace"),
                "bytes": list(byte_seq),
            }

            if token_id in merge_parents:
                id_a, id_b = merge_parents[token_id]
                entry["merges"] = [id_a, id_b]
                entry["merge_rank"] = merge_rank[token_id]
            else:
                entry["merges"] = None
                entry["merge_rank"] = None

            tokens[str(token_id)] = entry

        data = {
            "vocab_size": len(vocab),
            "tokens": tokens,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        LOGGER.info(f"Tokenizer saved to '{output_path}'.")

    except Exception as e:
        LOGGER.error(f"An error occurred while saving the tokenizer: {e}")


def init_training_pipeline() -> None:
    """Initializes the training pipeline by loading, reading, and processing the training data."""
    LOGGER.info("Initializing training pipeline...")

    # Step 1: Load books
    books_list, n_books = load_books(TRAINING_DATA_DIR)
    LOGGER.info(f"Loaded {n_books} book(s) for training: {books_list}")

    # Exit if no books were found
    if n_books == 0:
        LOGGER.error("No books found for training. Please add .txt files to the 'books' directory and try again.")
        return

    # Step 2: Read books and create corpus
    corpus = read_books(books_list, TRAINING_DATA_DIR)
    LOGGER.info(f"Combined corpus created with {len(corpus)} characters.")

    # Exit if corpus is empty after reading all books
    if not corpus:
        LOGGER.error("Corpus is empty after reading all books. Exiting pipeline.")
        return

    # Step 3: Build frequency table
    frequency_table = build_frequency_table(corpus)

    # Step 4: Train BPE tokenizer
    merges, vocab = train(frequency_table, VOCAB_SIZE)

    # Step 5: Save tokenizer to JSON
    save_tokenizer(merges, vocab, MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    init_training_pipeline()