from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field # type: ignore

class Step(BaseModel):
    name: str
    type: str
    query: Optional[str] = Field(default="")
    url: Optional[str] = Field(default="")
    urls: List[str] = Field(default_factory=list)
    source: Optional[str] = Field(default="")
    model_name: Optional[str] = Field(default="")
    prompt: Optional[str] = Field(default="")
    text: Optional[str] = Field(default="")
    result_path: Optional[str] = Field(default="")
    result_path: Optional[str] = Field(default="")
    input_text: Optional[str] = Field(default="")
    method: Optional[str] = Field(default="")
    payload: Dict[str, Any] = Field(default_factory=dict)
    search_type: str = Field(default="duckduckgo")
    num_results: Union[int, str] = Field(default=5)
    fine_tune_params: Dict[str, Any] = Field(default_factory=dict)
    input_steps: List[str] = Field(default_factory=list)
    audio_file: Optional[str] = Field(default="")
    image_params: Dict[str, Any] = Field(default_factory=dict)

class PipelineConfig(BaseModel):
    context: Dict[str, Any] = Field(default_factory=dict)
    steps: List[Step]