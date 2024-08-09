import os
import logging
from typing import Dict, Any, Optional

from langchain.tools import BaseTool
from pydantic import Field
from TTS.api import TTS
import speech_recognition as sr
from ollama import Client as OllamaClient
import openai

from config import Step

# Set up logging
logger = logging.getLogger(__name__)

class TextToSpeechTool(BaseTool):
    """Tool for converting text to speech."""

    name = "TextToSpeech"
    description = "Converts text to speech."
    tts_model: Optional[TTS] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self.tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            self.tts_model = None

    def _run(self, step: Step, context: Dict[str, Any]) -> str:
        if not self.tts_model:
            return "TTS model is not initialized."
        try:
            self.tts_model.tts_to_file(text=step.text, file_path=step.data_path)
            return f"Text-to-speech conversion completed. Audio saved to {step.data_path}"
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {e}")
            return f"An error occurred during text-to-speech conversion: {str(e)}"
    
    async def _arun(self, step: Step, context: Dict[str, Any]) -> str:
        return self._run(step, context)

class SpeechToTextTool(BaseTool):
    """Tool for converting speech to text."""

    name = "SpeechToText"
    description = "Converts speech to text using either offline (CMU Sphinx) or online (Google Speech Recognition) methods."
    stt_recognizer: sr.Recognizer = Field(default_factory=sr.Recognizer)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, step: Step, context: Dict[str, Any]) -> str:
        audio_file = step.audio_file
        method = step.method.lower() if hasattr(step, 'method') else 'offline'

        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.stt_recognizer.record(source)

            if method == 'offline':
                text = self._offline_recognition(audio)
            elif method == 'online':
                text = self._online_recognition(audio)
            else:
                return f"Invalid method specified: {method}. Use 'offline' or 'online'."

            return f"Speech-to-text conversion completed. Transcribed text: {text}"
        except Exception as e:
            logger.error(f"Error in speech-to-text conversion: {e}")
            return f"An error occurred during speech-to-text conversion: {str(e)}"
        
    async def _arun(self, step: Step, context: Dict[str, Any]) -> str:
        return self._run(step, context)

    def _offline_recognition(self, audio: sr.AudioData) -> str:
        try:
            return self.stt_recognizer.recognize_sphinx(audio)
        except sr.UnknownValueError:
            return "Sphinx could not understand the audio"
        except sr.RequestError as e:
            logger.error(f"Sphinx error: {e}")
            return f"Sphinx error; {e}"

    def _online_recognition(self, audio: sr.AudioData) -> str:
        try:
            return self.stt_recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand the audio"
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition error: {e}")
            return f"Could not request results from Google Speech Recognition service; {e}"

class InferenceTool(BaseTool):
    """Tool for running inference using various models."""

    name: str = "RunInference"
    description: str = "Runs inference on a given prompt using the loaded model."
    model: Optional[Any] = Field(default=None)
    model_name: str = Field(default="")
    model_source: str = Field(default="")

    class Config:
        arbitrary_types_allowed = True

    def load_model(self, source: str, model_name: str) -> str:
        if source == "ollama":
            self.model = OllamaClient()
            self.model_source = "ollama"
        elif source == "openai":
            openai.api_key = os.environ.get("OPENAI_API_KEY", "")
            if not openai.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.model = openai
            self.model_source = "openai"
        else:
            raise ValueError(f"Unknown model source: {source}")
        self.model_name = model_name
        return f"Model {model_name} loaded from {source}"

    def _run(self, step: Step, context: Dict[str, Any]) -> str:
        if self.model is None:
            try:
                self.load_model(step.source, step.model_name)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return f"Failed to load model: {str(e)}"

        try:
            if self.model_source == "ollama":
                response = self.model.generate(model=self.model_name, prompt=step.prompt)
                result = response['response']
            elif self.model_source == "openai":
                response = self.model.Completion.create(
                    engine=self.model_name,
                    prompt=step.prompt,
                    max_tokens=100
                )
                result = response.choices[0].text.strip()
            else:
                return "Unsupported model source for inference"

            if hasattr(step, 'result_path') and step.result_path:
                try:
                    with open(step.result_path, 'w') as f:
                        f.write(result)
                    logger.info(f"Inference result saved to {step.result_path}")
                except Exception as e:
                    logger.error(f"Failed to save inference result to {step.result_path}: {str(e)}")

            return result
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return f"An error occurred during inference: {str(e)}"
    
    async def _arun(self, step: Step, context: Dict[str, Any]) -> str:
        return self._run(step, context)