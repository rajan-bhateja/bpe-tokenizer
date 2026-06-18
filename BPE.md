# Byte Pair Encoding (BPE) Tokenizer - train.py Documentation

## Overview

`train.py` implements a **Byte Pair Encoding (BPE)** tokenizer trainer. BPE is a data compression algorithm that iteratively merges the most frequent pair of consecutive bytes/tokens into a single new token. This process is commonly used in large language models (like GPT-2/3) to create efficient vocabularies that balance vocabulary size with compression effectiveness.

## Algorithm Summary

BPE training follows these steps:

1. **Initialization**: Start with the 256 individual bytes as the initial vocabulary
2. **Frequency Analysis**: Count all adjacent pairs of tokens in the corpus
3. **Merge**: Find the most frequent pair and create a new token by merging them
4. **Repeat**: Update the corpus with the merged tokens and repeat steps 2-3 until the target vocabulary size is reached

This produces a hierarchical vocabulary where complex tokens are built from simpler ones, along with a "merge table" that records the order in which tokens were created.

---

## Core Functions

### `load_books(data_dir: str) -> tuple[list[str], int]`

**Purpose**: Discover and catalog training data files.

**Parameters**:
- `data_dir` (str): Directory path containing `.txt` book files

**Returns**: 
- Tuple of (list of book filenames, count of books)
- Returns `([], 0)` if directory doesn't exist or is empty

**Process**:
1. Validates that the specified directory exists
2. Scans for all files ending with `.txt`
3. Logs warnings if directory is empty
4. Returns both the list and count for convenience

**Example**: 
```
books_list = ["pride_and_prejudice.txt", "jane_eyre.txt"]
n_books = 2
```

---

### `read_books(books_list: list[str], data_dir: str) -> str`

**Purpose**: Load and concatenate all book contents into a single training corpus.

**Parameters**:
- `books_list` (list[str]): List of book filenames to read
- `data_dir` (str): Directory containing the book files

**Returns**: Single concatenated string containing all book contents

**Process**:
1. Iterates through each book in the list
2. Opens and reads each file with UTF-8 encoding
3. Skips empty books with a warning
4. Concatenates all content into a single corpus string
5. Logs the character count for each book

**Important**: Books are concatenated without separators, so you may want to ensure books end with natural breaks.

---

### `build_frequency_table(corpus: str) -> dict[tuple[int, ...], int]`

**Purpose**: Tokenize the corpus into chunks and build a frequency table of unique byte sequences.

**Parameters**:
- `corpus` (str): The raw text corpus to tokenize

**Returns**: Dictionary mapping byte sequence tuples to their frequency counts

**Key Details**:

**GPT-2 Regex Pattern**: The function uses this pattern:
```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

This pattern intelligently chunks text into:
- Contractions (`'s`, `'t`, `'ve`, etc.)
- Words with optional preceding punctuation
- Numbers (1-3 digits)
- Spaces and punctuation
- Newlines and whitespace

**Process**:
1. Uses `regex.findall()` to extract all chunks matching the GPT-2 pattern
2. Counts chunk frequencies using `Counter`
3. Converts each chunk to UTF-8 bytes and creates a tuple of byte values
4. Returns a dictionary where keys are byte tuples and values are frequencies

**Example**:
```python
# For corpus "hello hello world"
# Output might be:
{
    (104, 101, 108, 108, 111): 2,      # "hello" appears 2 times
    (119, 111, 114, 108, 100): 1,      # "world" appears 1 time
}
```

---

### `get_pair_counts(frequency_table: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]`

**Purpose**: Count the frequency of every adjacent pair of bytes across all sequences.

**Parameters**:
- `frequency_table` (dict): Frequency table of byte sequences from `build_frequency_table()`

**Returns**: Dictionary mapping (byte1, byte2) pairs to their total frequency

**Process**:
1. Iterates through each byte sequence in the frequency table
2. For each sequence, creates all adjacent pairs (sequence[i], sequence[i+1])
3. Multiplies each pair's count by the sequence's frequency
4. Accumulates counts for pairs that appear in multiple sequences

**Example**:
```python
# Frequency table: {(1,2,3): 5, (1,2,4): 3}
# Output:
{
    (1, 2): 8,    # appears in both sequences (5+3 total)
    (2, 3): 5,    # only in first sequence
    (2, 4): 3,    # only in second sequence
}
```

---

### `merge_sequence(sequence: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]`

**Purpose**: Replace all occurrences of a specific pair with a new token ID in a sequence.

**Parameters**:
- `sequence` (tuple[int, ...]): The byte sequence to process
- `pair` (tuple[int, int]): The pair to find and replace
- `new_id` (int): The new token ID to replace the pair with

**Returns**: Modified sequence with all pair occurrences replaced

**Process**:
1. Iterates through the sequence with a pointer
2. When the pair is found at the current position, replaces it with `new_id` and skips 2 positions
3. Otherwise, keeps the byte as-is and moves 1 position forward
4. Returns the modified sequence as a tuple

**Important**: This operation is non-overlapping (greedy). Once a pair is matched, the pointer jumps past it.

**Example**:
```python
merge_sequence((1,2,3,2,1), (1,2), 256)
# Returns: (256, 3, 2, 1)
# The pair (1,2) at position 0 is replaced with 256
```

---

### `train(frequency_table: dict[tuple[int, ...], int], vocab_size: int) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]`

**Purpose**: Execute the main BPE training loop to build the final vocabulary.

**Parameters**:
- `frequency_table` (dict): Initial frequency table of byte sequences
- `vocab_size` (int): Target vocabulary size (e.g., 3500)

**Returns**: Tuple of:
1. `merges` (dict): Maps each merge pair to the new token ID it created
2. `vocab` (dict): Maps token IDs to their byte representations

**Process**:

1. **Initialization**:
   - Starts with base vocabulary of 256 byte tokens (0-255)
   - `next_id` begins at 256 for new merged tokens
   - Calculates number of merges needed: `vocab_size - 256`

2. **Main Loop** (performs `vocab_size - 256` iterations):
   - Calls `get_pair_counts()` to find all adjacent pairs
   - Finds the most frequent pair
   - Creates a new token by merging the pair's bytes
   - Records the merge in the `merges` dictionary
   - Updates `vocab` with the new token
   - Updates `frequency_table` by merging all occurrences across all sequences
   - Logs progress every 100 merges

3. **Early Stopping**: Stops if no more pairs exist or pair count is 0

4. **Output**: Returns both the merge history and final vocabulary

**Key Variables**:
- `vocab`: Maps token_id (int) → bytes. Initially `{0: b'\x00', 1: b'\x01', ..., 255: b'\xff'}`
- `merges`: Maps (id_a, id_b) → new_id. Ordered by merge sequence
- `next_id`: Incremented each merge, becomes 256, 257, 258, ...

---

### `save_tokenizer(merges: dict[tuple[int, int], int], vocab: dict[int, bytes], output_path: str) -> None`

**Purpose**: Serialize the trained tokenizer to a JSON file for later use.

**Parameters**:
- `merges` (dict): Merge history from `train()`
- `vocab` (dict): Final vocabulary from `train()`
- `output_path` (str): Path where JSON file will be saved (e.g., "tokenizer.json")

**Returns**: None (saves to file)

**Output Format**:

The JSON file contains:
```json
{
  "vocab_size": 3500,
  "tokens": {
    "0": {
      "content": "\u0000",
      "bytes": [0],
      "merges": null,
      "merge_rank": null
    },
    "256": {
      "content": "he",
      "bytes": [104, 101],
      "merges": [104, 101],
      "merge_rank": 0
    },
    ...
  }
}
```

**Structure of each token entry**:
- `content`: Unicode representation of the token (with invalid bytes shown as replacement character)
- `bytes`: List of byte values
- `merges`: `[id_a, id_b]` if this token was created by merging, `null` for base bytes
- `merge_rank`: The order this merge was performed (0-indexed), `null` for base bytes

**Process**:
1. Builds a reverse lookup from `new_id` → original pair
2. Builds a rank lookup from `new_id` → merge order
3. Iterates through vocabulary, creating an entry for each token
4. Writes the complete structure to JSON with nice formatting

---

### `init_training_pipeline() -> None`

**Purpose**: Orchestrate the entire training pipeline from start to finish.

**Parameters**: None

**Returns**: None

**Execution Flow**:

1. **Load Books**: Calls `load_books(TRAINING_DATA_DIR)` to find training data
   - Returns early if no books found

2. **Read Corpus**: Calls `read_books()` to load all book contents
   - Returns early if corpus is empty

3. **Build Frequency Table**: Calls `build_frequency_table()` to tokenize and count
   - Creates initial byte frequency table

4. **Train Model**: Calls `train()` to perform BPE merges
   - Builds vocabulary and merge history

5. **Save Model**: Calls `save_tokenizer()` to serialize results
   - Saves to JSON file for later use

**Logging**: Each step logs progress and any warnings/errors.

---

## Configuration Constants

```python
TRAINING_DATA_DIR = "books"          # Directory containing .txt training files
VOCAB_SIZE = 3_500                   # Target vocabulary size (bytes 0-255 + merges)
MODEL_OUTPUT_PATH = "tokenizer.json" # Output file path
GPT2_REGEX_PATTERN = ...             # Tokenization pattern for chunking text
```

**Notes**:
- `VOCAB_SIZE` of 3,500 means 256 base bytes + 3,244 merged tokens
- All constants are defined at module level for easy modification

---

## Main Entry Point

```python
if __name__ == "__main__":
    init_training_pipeline()
```

When executed as a script, this runs the complete training pipeline with the configured constants.

---

## Example Usage

```bash
# Ensure books/ directory exists with .txt files
python train.py
```

**Expected Output**:
1. Logs indicating books loaded and corpus size
2. Periodic merge progress (every 100 merges)
3. Final statistics: vocabulary size and total merges
4. `tokenizer.json` file created

---

## Error Handling

The code uses a logging system (`LOGGER`) to report:
- **Errors**: Missing directories, file read errors, saving errors → exits/returns
- **Warnings**: Empty books, empty corpus, no more pairs → continues/stops gracefully

All functions include try-except blocks and appropriate error messages.

---

## Key Design Decisions

1. **Regex Tokenization**: Uses GPT-2 pattern for linguistically-aware chunking
2. **Tuple Representation**: Byte sequences stored as immutable tuples for use as dictionary keys
3. **Frequency Table Update**: Rebuilds entire frequency table after each merge (can be optimized)
4. **JSON Serialization**: Stores merge relationships for educational/debugging purposes
5. **Logging**: Comprehensive logging for monitoring training progress and debugging

---

## Performance Characteristics

- **Time Complexity**: O(n_merges × corpus_size) where each merge requires updating the frequency table
- **Space Complexity**: O(unique_sequences + vocabulary_size)
- **For 3,500 vocab on typical book corpus**: Takes seconds to minutes depending on corpus size

---

## Dependencies

- `os`: File system operations
- `json`: Serialization
- `regex`: Advanced tokenization with Unicode support
- `collections.Counter`: Efficient frequency counting
- `logger`: Custom logging module (project-specific)
