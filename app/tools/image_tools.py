from langchain.tools import BaseTool # type: ignore
from config import Step
from typing import Dict, Any, Optional, Union
from pydantic import Field # type: ignore
from diffusers import StableDiffusionPipeline, FluxPipeline # type: ignore
import torch # type: ignore
from PIL import Image  # Added this import
import logging
import os

logger = logging.getLogger(__name__)

class GenerateImageTool(BaseTool):
    name = "GenerateImage"
    description = "Generates an image from text using either Stable Diffusion or FLUX model."
    image_model: Optional[Union[StableDiffusionPipeline, FluxPipeline]] = Field(default=None)
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    model_type: str = Field(default="stable_diffusion")
    model_id: str = Field(default="runwayml/stable-diffusion-v1-5")

    class Config:
        arbitrary_types_allowed = True

    def _load_model(self) -> None:
        """Load the specified image generation model."""
        try:
            if self.model_type == "stable_diffusion":
                self.image_model = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.image_model.to(self.device)
                if self.device == "cuda":
                    self.image_model.enable_attention_slicing()
            elif self.model_type == "flux":
                self.image_model = FluxPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                if self.device == "cuda":
                    self.image_model.to(self.device)
                else:
                    self.image_model.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            logger.info(f"Model {self.model_id} of type {self.model_type} loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {str(e)}")
            raise

    def clear_gpu_memory(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _run(self, step: Step, context: Dict[str, Any]) -> str:
        try:
            logger.info(f"Generating image with prompt: {step.prompt}")
            if self.image_model is None:
                self.model_type = step.image_params.get("model_type", self.model_type)
                self.model_id = step.image_params.get("model_id", self.model_id)
                logger.info(f"Loading {self.model_type} model...")
                self._load_model()
                logger.info("Image model loaded successfully")

            logger.info("Generating image...")
            if self.model_type == "stable_diffusion":
                image = self.image_model(
                    prompt=step.prompt,
                    num_inference_steps=step.image_params.get("num_inference_steps", 50),
                    guidance_scale=step.image_params.get("guidance_scale", 7.5)
                ).images[0]
            elif self.model_type == "flux":
                image = self.image_model(
                    step.prompt,
                    guidance_scale=step.image_params.get("guidance_scale", 4.0),
                    num_inference_steps=step.image_params.get("num_inference_steps", 4),
                    max_sequence_length=step.image_params.get("max_sequence_length", 256),
                    generator=torch.Generator(self.device).manual_seed(step.image_params.get("seed", 0))
                ).images[0]
            
            if image is None or not isinstance(image, Image.Image):
                raise ValueError("Failed to generate image")

            logger.info("Image generated successfully")

            output_dir = os.path.dirname(step.result_path)
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving image to {step.result_path}")
            image.save(step.result_path)
            logger.info("Image saved successfully")

            self.clear_gpu_memory()

            return f"Image generated using {self.model_type} and saved to {step.result_path}"
        except Exception as e:
            logger.error(f"Error in GenerateImageTool: {str(e)}")
            self.clear_gpu_memory()
            return f"Error generating image: {str(e)}"

    async def _arun(self, step: Step, context: Dict[str, Any]) -> str:
        """Asynchronous version of _run method."""
        return self._run(step, context)