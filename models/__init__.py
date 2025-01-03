from pydantic import BaseModel


class LLM(BaseModel):
    pretrained_model_path: str
    train_dataset_path: str


class Prompt(BaseModel):
    user_prompt: str
