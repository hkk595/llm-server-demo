from fastapi import FastAPI, BackgroundTasks
from controllers.llm import llm_fine_tune, llm_prompt
from models import LLM, Prompt

import os
output_path = os.getenv("OUTPUT_PATH", "./output")

app = FastAPI()

# Global variables
app.model_trained = False
app.model = None
app.tokenizer = None


@app.post("/fine-tune")
async def fine_tune(llm: LLM, background_tasks: BackgroundTasks):
    app.model_trained = False
    background_tasks.add_task(llm_fine_tune, llm.pretrained_model, llm.train_dataset, output_path, app)
    return {"status": "The model is being trained."}


@app.post("/prompt")
async def prompt_respond(prompt: Prompt):
    if not app.model or not app.tokenizer:
        return {
            "prompt": prompt.user_prompt,
            "response": "",
            "error": "The model is not ready."
        }
    response = llm_prompt(prompt.user_prompt, app.model, app.tokenizer)
    return {
        "prompt": prompt.user_prompt,
        "response": response,
        "error": None
    }


@app.get("/check")
async def check_status():
    model_status = "ready" if app.model_trained else "not ready"
    return {"status": f"The model is {model_status}."}
