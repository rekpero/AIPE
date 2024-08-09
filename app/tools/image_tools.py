from langchain.tools import BaseTool # type: ignore
from config import Step
from typing import Dict, Any, Optional
from pydantic import Field # type: ignore
from diffusers import StableDiffusionPipeline # type: ignore
import torch # type: ignore
import logging
import os

logger = logging.getLogger(__name__)

class GenerateImageTool(BaseTool):
    name = "GenerateImage"
    description = "Generates an image from text, loading the model if necessary."
    image_model: Optional[StableDiffusionPipeline] = Field(default=None)
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    model_id: str = Field(default="runwayml/stable-diffusion-v1-5")

    class Config:
        arbitrary_types_allowed = True

    def _load_model(self) -> None:
        """Load the Stable Diffusion model."""
        try:
            self.image_model = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.image_model.to(self.device)
            if self.device == "cuda":
                self.image_model.enable_attention_slicing()
            logger.info(f"Model {self.model_id} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {str(e)}")
            raise

    def _run(self, step: Step, context: Dict[str, Any]) -> str:
        """Generate an image from a text prompt."""
        try:
            logger.info(f"Generating image with prompt: {step.prompt}")
            if self.image_model is None:
                logger.info("Loading image model...")
                self._load_model()
                logger.info("Image model loaded successfully")

            logger.info("Generating image...")
            image = self.image_model(
                prompt=step.prompt,
                num_inference_steps=step.image_params.get("num_inference_steps", 50),
                guidance_scale=step.image_params.get("guidance_scale", 7.5)
            ).images[0]
            logger.info("Image generated successfully")

            output_dir = os.path.dirname(step.data_path)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving image to {step.data_path}")
            image.save(step.data_path)
            logger.info("Image saved successfully")

            return f"Image generated and saved to {step.data_path}"
        except Exception as e:
            logger.error(f"Error in GenerateImageTool: {str(e)}")
            return f"Error generating image: {str(e)}"

    async def _arun(self, step: Step, context: Dict[str, Any]) -> str:
        """Asynchronous version of _run method."""
        return self._run(step, context)

    def clear_gpu_memory(self) -> None:
        """Clear GPU memory and unload the model."""
        if self.image_model is not None:
            del self.image_model
            self.image_model = None
        if self.device == "cuda":
            with GPUMemoryManager():
                empty_cache()
        logger.info("GPU memory cleared and model unloaded")

class GPUMemoryManager:
    """Context manager for GPU memory management."""
    def __enter__(self):
        torch.cuda.empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()