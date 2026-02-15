from pydantic import BaseModel


class Evaluation(BaseModel):
    is_accepted: bool
    feedback: str
