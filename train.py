import os
import json
import uuid
from fastapi import UploadFile, APIRouter, FastAPI, Form, File
from datetime import datetime
from typing import Any, Annotated

from logger import get_logger


# CONSTANTS
DATA_DIR = "data_dir"


# Set up logging
LOGGER = get_logger(__name__)


def validate_file_extension(filename: str) -> bool:
    """Validates whether a received file is a `.txt` or not. Returns a boolean value."""
    if filename.endswith(".txt"):
        LOGGER.info(f"File '{filename}' is a valid '.txt' file.")
        return True
    LOGGER.error(f"File '{filename}' is not a '.txt' file.")
    return False


def generate_uuid() -> str:
    """Returns a string `UUID4` value"""
    return str(uuid.uuid4())


def create_new_file_name(filename: str, timestamp: str) -> str:
    """Creates and returns a new file name using `timestamp` and `uuid`"""
    base_name = filename.replace(".txt", "")
    generated_uuid = generate_uuid()
    new_file_name = f"{generated_uuid}_{timestamp}_{base_name}.txt"
    return new_file_name


app = FastAPI()


@app.post("/v1/train")
async def train(
        username: Annotated[str, Form()],
        files: Annotated[list[UploadFile], File()]
    )-> dict[str, Any]:
    """Train a custom BPE tokenizer using the provided `.txt` file and save the trained tokenizer to disk."""

    valid_files: list[str] = []
    rejected_files: list[str] = []

    # Current timestamp for each file
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    
    for file in files:
        file_name = file.filename

        if file_name:
            is_valid_file_extension = validate_file_extension(file_name)

            # If file is a valid .txt
            if is_valid_file_extension:
                
                # Read the file contents
                LOGGER.info(f"Reading '{file_name}'...")
                content = await file.read()
                if not content:
                    LOGGER.warning(f"File '{file_name}' appears to be empty. Skipping and adding to rejected files list")
                    rejected_files.append(file_name)
                    continue

                # Get the new file name in the required format
                os.makedirs(DATA_DIR, exist_ok = True)
                new_file_name = create_new_file_name(file_name, timestamp)
                new_file_path = os.path.join(DATA_DIR, new_file_name)

                # Write the file contents to the data directory
                with open(new_file_path, "wb") as f:
                    n_bytes = f.write(content)
                    if n_bytes > 0:
                        LOGGER.info(f"Written {n_bytes} characters to '{new_file_path}'")
                        LOGGER.info(f"Appending '{file_name}' to valid files list")
                        valid_files.append(file_name)
            
            # If file is not a valid .txt
            else:
                LOGGER.warning(f"Appending '{file_name}' to rejected files list")
                rejected_files.append(file_name)

    response = {
        "user": username,
        "uploaded_files": [f.filename for f in files],
        "valid_files": valid_files,
        "rejected_files": rejected_files,
        "timestamp": datetime.now().isoformat()
    }
    LOGGER.info(f"API Response:\n{json.dumps(response, indent=2)}")
    return response