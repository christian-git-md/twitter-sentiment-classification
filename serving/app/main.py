from asyncio import Queue

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import asyncio
import logging

from transformers import pipeline

from model.dataset import clean_text

app = FastAPI()
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def start_model_server_loop():
    """
    Start the model server loop upon application startup.

    Roughly mimicking the Starlette setup from https://huggingface.co/docs/transformers/main/en/pipeline_webserver.
    """
    q = asyncio.Queue()
    app.state.model_queue = q
    asyncio.create_task(model_server_loop(q))


@app.post("/")
@app.post("/evaluate_text_sentiment")
async def evaluate_text_sentiment(request: Request):
    """
    Evaluate the sentiment of the provided text. This endpoint accepts text data as raw payload, processes it through
    the sentiment analysis model, and returns the sentiment.
    """
    payload = await request.body()
    string = payload.decode("utf-8")
    response_q = asyncio.Queue()
    await request.app.state.model_queue.put((string, response_q))
    return await response_q.get()


async def model_server_loop(in_queue: Queue):
    """
    A simple server loop, so requests can be processed sequentially by the model.
    """
    model = pipeline(
        task="sentiment-analysis",
        model="christian-git-md/distilbert-base-uncased-finetuned-twitter-noleak",
    )
    while True:
        string, response_queue = await in_queue.get()
        try:
            string = clean_text(string)
            out = model(string)
            await response_queue.put(JSONResponse(content=out))
        except (KeyboardInterrupt, SystemExit):
            exit()
        except Exception:
            logger.exception("An error occurred during model processing.")
            await response_queue.put(
                JSONResponse(
                    status_code=500, content={"error": "Internal server error"}
                )
            )


# uvicorn serving.app.main:app --host 0.0.0.0 --port 9090
