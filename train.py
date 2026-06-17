import os
import regex
from collections import Counter
from logger import get_logger


LOGGER = get_logger(__name__)


# CONSTANTS
TRAINING_DATA_DIR = "books"
GPT2_REGEX_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


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
    LOGGER.info(f"Top 10 chunks: {chunks_count.most_common(10)}")

    frequency_table = {
        tuple(chunk.encode("utf-8")): count
        for chunk, count in chunks_count.items()
    }

    LOGGER.info(f"Total chunks processed: {sum(chunks_count.values())}")
    LOGGER.info(f"Unique byte sequences: {len(frequency_table)}")
    return frequency_table


def init_training_pipeline() -> None:
    """Initializes the training pipeline by loading, reading, and processing the training data."""
    LOGGER.info("Initializing training pipeline...")

    books_list, n_books = load_books(TRAINING_DATA_DIR)
    LOGGER.info(f"Loaded {n_books} book(s) for training: {books_list}")

    if n_books == 0:
        LOGGER.error("No books to process. Exiting pipeline.")
        return

    corpus = read_books(books_list, TRAINING_DATA_DIR)
    LOGGER.info(f"Combined corpus created with {len(corpus)} characters.")

    if not corpus:
        LOGGER.error("Corpus is empty after reading all books. Exiting pipeline.")
        return

    frequency_table = build_frequency_table(corpus)


if __name__ == "__main__":
    init_training_pipeline()