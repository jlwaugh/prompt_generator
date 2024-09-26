# Prompt Generator

This is a "Napthafied" agent, converted from an equivalent found in the [Swarms](https://swarms.world) marketplace. Given a simple description, it can refine or generate prompts.

### Original Code

```
import os
from typing import List

from loguru import logger
from pydantic import BaseModel, Field

from swarms import OpenAIFunctionCaller, create_file_in_folder


class PromptUseCase(BaseModel):
    title: str = Field(
        ...,
        description="The name of the use case.",
    )
    description: str = Field(
        ...,
        description="The description of the use case.",
    )


class PromptSchema(BaseModel):
    name: str = Field(
        ...,
        description="The name of the prompt.",
    )
    prompt: str = Field(
        ...,
        description="The prompt to generate the response.",
    )
    description: str = Field(
        ...,
        description="The description of the prompt.",
    )
    tags: str = Field(
        ...,
        description="The tags for the prompt denoted by a comma sign: Code Gen Prompt, Pytorch Code Gen Agent Prompt, Finance Agent Prompt, ",
    )
    useCases: List[PromptUseCase] = Field(
        ...,
        description="The use cases for the prompt.",
    )


class PromptGeneratorAgent:
    """
    A class that generates prompts based on given tasks and publishes them to the marketplace.

    Args:
        system_prompt (str, optional): The system prompt to use. Defaults to None.
        max_tokens (int, optional): The maximum number of tokens in the generated prompt. Defaults to 1000.
        temperature (float, optional): The temperature value for controlling randomness in the generated prompt. Defaults to 0.5.
        schema (BaseModel, optional): The base model schema to use. Defaults to PromptSchema.

    Attributes:
        llm (OpenAIFunctionCaller): An instance of the OpenAIFunctionCaller class for making function calls to the OpenAI API.

    Methods:
        clean_model_code: Cleans the model code by removing extra escape characters, newlines, and unnecessary whitespaces.
        upload_to_marketplace: Uploads the generated prompt data to the marketplace.
        run: Creates a prompt based on the given task and publishes it to the marketplace.
    """

    def __init__(
        self,
        system_prompt: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.5,
        schema: BaseModel = PromptSchema,
    ):
        self.llm = OpenAIFunctionCaller(
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            base_model=schema,
            parallel_tool_calls=False,
        )

    def clean_model_code(self, model_code_str: str) -> str:
        """
        Cleans the model code by removing extra escape characters, newlines, and unnecessary whitespaces.

        Args:
            model_code_str (str): The model code string to clean.

        Returns:
            str: The cleaned model code.
        """
        cleaned_code = model_code_str.replace("\\n", "\n").replace(
            "\\'", "'"
        )
        cleaned_code = cleaned_code.strip()
        return cleaned_code

    def upload_to_marketplace(self, data: dict) -> dict:
        """
        Uploads the generated prompt data to the marketplace.

        Args:
            data (dict): The prompt data to upload.

        Returns:
            dict: The response from the marketplace API.
        """
        import requests
        import json

        url = "https://swarms.world/api/add-prompt"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('SWARMS_API_KEY')}",
        }
        response = requests.post(
            url, headers=headers, data=json.dumps(data)
        )
        return str(response.json())

    def run(self, task: str) -> str:
        """
        Creates a prompt based on the given task and publishes it to the marketplace.

        Args:
            task (str): The task description for generating the prompt.

        Returns:
            dict: The response from the marketplace API after uploading the prompt.
        """
        out = self.llm.run(task)
        name = out["name"]
        logger.info(f"Prompt generated: {out}")

        create_file_in_folder(
            "auto_generated_prompts", f"prompt_{name}.json", str(out)
        )
        logger.info(f"Prompt saved to file: prompt_{name}.json")

        # Clean the model code
        prompt = out["prompt"]
        description = out["description"]
        tags = out["tags"]
        useCases = out["useCases"]

        data = {
            "name": name,
            "prompt": self.clean_model_code(prompt),
            "description": description,
            "tags": tags,
            "useCases": useCases,
        }

        create_file_in_folder(
            "auto_generated_prompts",
            f"prompt_{name}_cleaned.json",
            str(data),
        )

        # Now submit to swarms API
        logger.info("Uploading to marketplace...")
        return self.upload_to_marketplace(data)


# Example usage:
agent = PromptGeneratorAgent()
response = agent.run(
    "Create a prompt for an agent that is really good for email greeting, make sure the agent doesn't sound like a robot or an AI. Provide many-shot examples and instructions for the agent to follow."
)
```

### Try it out!

1. Clone

```bash
git clone https://github.com/jlwaugh/prompt_generator && cd prompt_generator
```

2. Install

```bash
poetry install
```

3. Run

```bash
poetry run python prompt_generator/run.py
```
