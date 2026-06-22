import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Annotated
from logger import get_logger

from decoder import load_vocab, decode as decode_tokens


# CONSTANTS
TOKENIZER_MODEL_PATH = "tokenizer.json"

with open(TOKENIZER_MODEL_PATH, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

N_VOCAB = int(tokenizer_data["vocab_size"] - 1)


# Set up logging
LOGGER = get_logger(__name__)


# Classes for input validation
class EncodeRequest(BaseModel):
    """Request model for encoding text."""
    text: Annotated[str, Field(description="Text string to encode", min_length=1)]


class DecodeRequest(BaseModel):
    """Request model for decoding token IDs."""
    token_ids: list[Annotated[int, Field(description="List of token IDs to decode", ge=0, le=N_VOCAB)]]
   

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI app"""
    LOGGER.info("Starting FastAPI app...")
    app.state.vocab = load_vocab(TOKENIZER_MODEL_PATH)
    if not app.state.vocab:
        LOGGER.error("Failed to load tokenizer. Shutting down...")
        return
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
    LOGGER.info(f"Decoding {len(decode_request.token_ids)} token(s): {decode_request.token_ids}")

    # vocab = load_vocab(TOKENIZER_MODEL_PATH)
    result = decode_tokens(decode_request.token_ids, app.state.vocab)

    response = {
        "input": decode_request.token_ids,
        "output": result,
        "timestamp": datetime.now().isoformat()
    }
    LOGGER.info(f"API response:\n{json.dumps(response, indent=2)}")
    return response
