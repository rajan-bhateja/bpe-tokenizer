import json
from logger import get_logger


# Set up logging
LOGGER = get_logger(__name__)


# CONSTANTS
TOKENIZER_PATH = "tokenizer.json"


def load_vocab(tokenizer_path: str) -> dict[int, bytes]:
    """Loads the tokenizer JSON and builds an id --> bytes lookup for decoding."""
    try:
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab: dict[int, bytes] = {
            int(token_id): bytes(entry["bytes"])
            for token_id, entry in data["tokens"].items()
        }

        LOGGER.info(f"Loaded vocab from '{tokenizer_path}'. Vocab size: {len(vocab)}")
        return vocab

    except FileNotFoundError:
        LOGGER.error(f"Tokenizer file not found at '{tokenizer_path}'.")
        raise

    except json.JSONDecodeError:
        LOGGER.error(f"Tokenizer file '{tokenizer_path}' is not a valid JSON.")
        raise

    except Exception as e:
        LOGGER.error(f"An error occurred while loading the vocab: {e}")
        raise


def decode(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    """Decodes a list of token IDs back into the original text string."""
    if not token_ids:
        LOGGER.warning("Received an empty token ID list. Returning empty string ('').")
        return ""

    byte_chunks: list[bytes] = []
    for token_id in token_ids:
        if token_id not in vocab:
            LOGGER.error(f"Unknown token ID encountered during decoding: {token_id}")
            raise ValueError(f"Unknown token ID: {token_id}")

        byte_chunks.append(vocab[token_id])

    # Concatenate all bytes first, then decode once.
    # This correctly handles multi-byte UTF-8 characters that may have been
    # split across two separate tokens during encoding.
    full_bytes = b"".join(byte_chunks)
    text = full_bytes.decode("utf-8", errors="replace")

    return text


def init_decoding_pipeline() -> None:
    """Main function to run the decoder interactively."""
    vocab = load_vocab(TOKENIZER_PATH)
    while True:
        sample_ids_input = input("Enter a list of token IDs (comma-separated), or 'exit' to quit: ")
        if sample_ids_input.lower().strip() == "exit":
            break
        sample_ids = [int(x.strip()) for x in sample_ids_input.split(",")]
        result = decode(sample_ids, vocab)
        LOGGER.info(f"Decoded {sample_ids} -> {result!r}")


if __name__ == "__main__":
    init_decoding_pipeline()
