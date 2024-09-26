from naptha_sdk.utils import get_logger, load_yaml
from typing import Dict, Any
from openai import OpenAI
import os

logger = get_logger(__name__)

class PromptGeneratorAgent:
    def __init__(self, system_prompt=None, max_tokens=1000, temperature=0.5):
        self.system_prompt = system_prompt or "You are an AI assistant specialized in generating high-quality prompts for various tasks."
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI()

    def generate_prompt(self, task: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        return response.choices[0].message.content

def run(inputs: Dict[str, Any], worker_nodes=None, orchestrator_node=None, flow_run=None, cfg=None) -> Dict[str, Any]:
    logger.info(f"Inputs: {inputs}")

    system_prompt = cfg.get('inputs', {}).get('system_prompt')

    agent = PromptGeneratorAgent(system_prompt=system_prompt)
    generated_prompt = agent.generate_prompt(inputs['prompt'])

    return {"generated_prompt": generated_prompt}

if __name__ == "__main__":
    cfg_path = "prompt_generator/component.yaml"
    cfg = load_yaml(cfg_path)
    inputs = {"prompt": "Create a prompt for an agent that is really good for email greeting, make sure the agent doesn't sound like a robot or an AI. Provide many-shot examples and instructions for the agent to follow."}
    result = run(inputs, cfg=cfg)
    print(result)
