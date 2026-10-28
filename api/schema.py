from pydantic import BaseModel, Field


class PDFRequest(BaseModel):
    path: str
    
class QuestionRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=20)


class SourceReference(BaseModel):
    citation_id: int
    source: str
    page: int
    distance: float


class QuestionResponse(BaseModel):
    answer: str
    sources: list[SourceReference]

