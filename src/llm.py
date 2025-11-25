import os
import base64
import requests
from openai import OpenAI

class OpenRouterClient:
    def __init__(self, config):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.text_model = config["models"]["text_model"]
        self.vision_model = config["models"]["vision_model"]

    def generate_diagram_code(self, prompt, system_prompt):
        """Generates Mermaid code from a text prompt."""
        response = self.client.chat.completions.create(
            model=self.text_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # Deterministic for code generation
        )
        content = response.choices[0].message.content
        # Strip markdown if present (fallback cleanup)
        if "```mermaid" in content:
            content = content.split("```mermaid")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return content

    def extract_prompt_from_image(self, image_path, system_prompt):
        """Extracts a text prompt from a diagram image."""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_string}"
                            },
                        }
                    ],
                }
            ],
            temperature=0.7, # Allow some creativity/variation in description
        )
        return response.choices[0].message.content
