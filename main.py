import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import Any
from logger import get_logger


# CONSTANTS
TOKENIZER_MODEL_PATH = "tokenizer.json"


# Set up logging
LOGGER = get_logger(__name__)


# Classes for input validation
class EncodeRequest(BaseModel):
    """Request model for encoding text."""
    text: str


class DecodeRequest(BaseModel):
    """Request model for decoding token IDs."""
    token_ids: list[int]


async def load_tokenizer() -> None:
    """Loads the trained BPE tokenizer from the JSON file."""
    try:
        global tokenizer
        tokenizer = None
        if TOKENIZER_MODEL_PATH:
            LOGGER.info(f"Loading tokenizer from '{TOKENIZER_MODEL_PATH}'...")
            with open(TOKENIZER_MODEL_PATH, "r", encoding="utf-8") as f:
                tokenizer_data = json.load(f)
                app.state.tokenizer = tokenizer_data
            LOGGER.info("Tokenizer loaded successfully.")

    except FileNotFoundError:
        LOGGER.error(f"Tokenizer file '{TOKENIZER_MODEL_PATH}' not found. Please train the tokenizer first.")
    except json.JSONDecodeError:
        LOGGER.error(f"Tokenizer file '{TOKENIZER_MODEL_PATH}' is not a valid JSON. Please check the file.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI app"""
    LOGGER.info("Starting FastAPI app...")
    await load_tokenizer()
    yield
    LOGGER.info("Shutting down FastAPI app...")


app = FastAPI(lifespan=lifespan)


@app.post("/v1/encode")
async def encode(encode_request: EncodeRequest) -> dict[str, Any]:
    """Encodes the input text using the trained BPE tokenizer and returns a list of token IDs."""
    LOGGER.info(f"Text received for encoding: {encode_request.text}")
    response = {
        "input": encode_request.text,
        "output": [],
        "timestamp": datetime.now().isoformat()
    }
    LOGGER.info(f"API response:\n{json.dumps(response, indent=2)}")
    return response


@app.post("/v1/decode")
async def decode(decode_request: DecodeRequest) -> dict[str, Any]:
    """Decodes a list of token IDs back into the original text using the trained BPE tokenizer."""
    LOGGER.info(f"Token IDs received for decoding: {decode_request.token_ids}")
    response = {
        "input": decode_request.token_ids,
        "output": "",
        "timestamp": datetime.now().isoformat()
    }
    LOGGER.info(f"API response:\n{json.dumps(response, indent=2)}")
    return response
