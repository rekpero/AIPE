import os
import yaml
import logging
import asyncio
import re
from typing import Dict, Any, List, Union, Optional

import torch
from pydantic import BaseModel, Field
from langchain.llms.base import BaseLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import AgentAction, AgentFinish, LLMResult, Generation
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from ollama import Client as OllamaClient

from config import PipelineConfig, Step
from tools.web_tools import WebSearchTool, WebScrapeTool
from tools.nlp_tools import TextToSpeechTool, SpeechToTextTool, InferenceTool
from tools.image_tools import GenerateImageTool
from tools.webhook_tool import WebhookTool

# Set up logging
logger = logging.getLogger(__name__)

class LlamaLLM(BaseLLM):
    """Custom LLM class for Llama model."""

    model_name: str = Field(default="llama2")
    client: Optional[OllamaClient] = None
    device: str = "cpu"
    
    def __init__(self, context: Dict[str, Any]):
        super().__init__()
        self.model_name = context.get('model_name', 'llama2')
        self.client = self._init_client()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        logger.info(f"Using model: {self.model_name}")

    def _init_client(self) -> Optional[OllamaClient]:
        max_retries = 20
        retry_interval = 10
        for i in range(max_retries):
            try:
                logger.info(f"Attempting to connect to Ollama (Attempt {i+1}/{max_retries})...")
                client = OllamaClient()
                client.list()
                logger.info("Successfully connected to Ollama")
                return client
            except Exception as e:
                if i < max_retries - 1:
                    logger.warning(f"Failed to connect to Ollama: {str(e)}")
                    logger.info(f"Retrying in {retry_interval} seconds...")
                    asyncio.sleep(retry_interval)
                else:
                    logger.error(f"Failed to connect to Ollama after {max_retries} attempts.")
                    raise

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        if self.client is None:
            raise ValueError("OllamaClient is not initialized")
        try:
            logger.info("Sending prompt to Ollama for inference...")
            response = self.client.generate(model=self.model_name, prompt=prompt)
            logger.info("Received response from Ollama")
            return response['response']
        except Exception as e:
            logger.error(f"Error during Ollama inference: {str(e)}")
            return "Error: Unable to perform inference with Ollama"

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> LLMResult:
        return LLMResult(generations=[[Generation(text=self._call(prompt, stop, run_manager))] for prompt in prompts])

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None) -> LLMResult:
        return LLMResult(generations=[[Generation(text=await asyncio.to_thread(self._call, prompt, stop, run_manager))] for prompt in prompts])

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "custom_llama"

class PipelinePromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\nThought: I now know the result of the action. I should use this information to decide on my next action.\n\n"
        
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

    def _get_prompt_dict(self):
        return {
            "template": self.template,
            "tools": self.tools,
        }

class PipelineEngine:
    """Main class for executing the AI pipeline."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        self.config = PipelineConfig(**config_data['pipeline'])
        self.context: Dict[str, Any] = self.config.context.copy()
        self.step_outputs: Dict[str, Any] = {}
        self.tools = self.create_tools()
        self.agent = self.create_agent()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Pipeline running on: {self.device}")

    def create_tools(self) -> List[Tool]:
        """Create and return a list of tools used in the pipeline."""
        return [
            Tool(name="WebSearch", func=WebSearchTool()._arun, description="Useful for searching the web for information."),
            Tool(name="WebScrape", func=WebScrapeTool()._arun, description="Useful for scraping content from given URLs."),
            Tool(name="RunInference", func=InferenceTool()._arun, description="Runs inference on a given prompt using the loaded model."),
            Tool(name="TextToSpeech", func=TextToSpeechTool()._arun, description="Converts text to speech."),
            Tool(name="SpeechToText", func=SpeechToTextTool()._arun, description="Converts speech to text."),
            Tool(name="GenerateImage", func=GenerateImageTool()._arun, description="Generates an image from text."),
            Tool(name="CallWebhook", func=WebhookTool()._arun, description="Calls a webhook with given parameters.")
        ]

    def create_agent(self) -> AgentExecutor:
        """Create and return an agent executor."""
        template = """
            You are an AI assistant tasked with executing a pipeline of steps efficiently. You have access to the following tools:

            {tools}

            Use these tools to complete the pipeline steps. Always use the tools in the order specified in the pipeline steps.

            Pipeline steps:
            {pipeline_steps}

            Current context:
            {current_context}

            Use the following format:

            Thought: Consider the current step and decide which tool to use
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I have completed all steps in the pipeline
            Final Answer: A summary of all actions taken and their results

            Begin!

            {agent_scratchpad}

            Thought: Let's start with the first step in the pipeline
        """

        prompt = PipelinePromptTemplate(
            template=template,
            tools=self.tools,
            input_variables=["pipeline_steps", "current_context", "intermediate_steps"]
        )

        class PipelineOutputParser(AgentOutputParser):
            def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                if "Final Answer:" in llm_output:
                    return AgentFinish(
                        return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                        log=llm_output,
                    )
                logger.info(f"LLM output: {llm_output}")
                
                match = re.search(r"Action: (.*?)[\n]*Action Input:[\s]*(.*)", llm_output, re.DOTALL)
                if not match:
                    raise ValueError(f"Could not parse LLM output: `{llm_output}`")
                action = match.group(1).strip()
                action_input = match.group(2)
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

        llm = LlamaLLM(context=self.context)
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        tool_names = [tool.name for tool in self.tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=PipelineOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )

        return AgentExecutor.from_agent_and_tools(agent=agent, tools=self.tools, verbose=True)

    def resolve_bindings(self, value: Any) -> Any:
        """Resolve context and step output bindings in the given value."""
        if isinstance(value, str):
            # Resolve context bindings
            context_pattern = r'\{context\.(\w+)\}'
            value = re.sub(context_pattern, lambda m: str(self.context.get(m.group(1), m.group(0))), value)
            
            # Resolve step output bindings
            step_pattern = r'\{steps\.(\w+)\}'
            value = re.sub(step_pattern, lambda m: str(self.step_outputs.get(m.group(1), m.group(0))), value)
        
        elif isinstance(value, dict):
            return {k: self.resolve_bindings(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.resolve_bindings(item) for item in value]
        
        return value

    def resolve_step(self, step: Step) -> Step:
        """Resolve bindings in the given step."""
        resolved_dict = {k: self.resolve_bindings(v) for k, v in step.dict().items()}
        return Step(**resolved_dict)

    async def run_step(self, step: Step) -> Any:
        """Run a single step of the pipeline."""
        resolved_step = self.resolve_step(step)
        logger.info(f"Running step: {resolved_step.name} (Type: {resolved_step.type})")
        logger.debug(f"Resolved step details: {resolved_step}")

        tool = next((t for t in self.tools if t.name == resolved_step.type), None)
        if tool is None:
            raise ValueError(f"Unknown step type: {resolved_step.type}")
        
        try:
            result = await tool.func(resolved_step, self.context)
            result_summary = str(result)[:1000] + "..." if len(str(result)) > 1000 else str(result)
            logger.info(f"Step {resolved_step.name} completed. Result length: {len(str(result))}")
            logger.debug(f"Step {resolved_step.name} result summary: {result_summary}")
            return result
        except Exception as e:
            logger.error(f"Error executing step {resolved_step.name}: {str(e)}")
            raise

    async def run(self) -> None:
        """Run the entire pipeline."""
        pipeline_steps = "\n".join([f"- {step.name}: {step.type}" for step in self.config.steps])
        result = await self.agent.arun(pipeline_steps=pipeline_steps, current_context=str(self.context))
        logger.info(f"Pipeline execution result: {result}")

        for step in self.config.steps:
            try:
                output = await self.run_step(step)
                self.step_outputs[step.name] = output
                self.context[step.name] = output
                logger.info(f"Step {step.name} output added to context and step_outputs")
                logger.debug(f"Current context after step {step.name}: {self.context}")
            except Exception as e:
                logger.error(f"Error in step {step.name}: {str(e)}")
                # Decide how to handle step failures (e.g., continue with next step or abort)

        self.clear_gpu_memory()

    def clear_gpu_memory(self) -> None:
        """Clear GPU memory after pipeline execution."""
        for tool in self.tools:
            if isinstance(tool, GenerateImageTool):
                tool.clear_gpu_memory()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("GPU memory cleared after pipeline execution")