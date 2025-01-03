from pydantic import BaseModel


class LLM(BaseModel):
    pretrained_model: str
    train_dataset: str


class Prompt(BaseModel):
    user_prompt: str
