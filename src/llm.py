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
        
        # Generation params
        params = config["models"].get("generation_params", {})
        self.text_temp = params.get("text_temperature", 0.0)
        self.vision_temp = params.get("vision_temperature", 0.7)

    def generate_diagram_code(self, prompt, system_prompt):
        """Generates Mermaid code from a text prompt.
        Returns: (content, usage)
        """
        response = self.client.chat.completions.create(
            model=self.text_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.text_temp, 
        )
        content = response.choices[0].message.content
        usage = response.usage
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        } if usage else {}

        # Strip markdown if present (fallback cleanup)
        if "```mermaid" in content:
            content = content.split("```mermaid")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return content, usage_dict

    def extract_prompt_from_image(self, image_path, system_prompt):
        """Extracts a prompt description from a diagram image."""
        
        # Read image and encode to base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Analyze this architecture diagram and describe it as a prompt."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}
                    ]
                }
            ],
            temperature=self.vision_temp,
        )
        
        usage = response.usage
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        } if usage else {}
        
        return response.choices[0].message.content, usage_dict

    def fix_diagram_code(self, original_code, error_message):
        """Attempts to fix Mermaid code that failed to render."""
        
        system_prompt = (
            "You are a Mermaid.js debugging expert. "
            "Your task is to fix the provided Mermaid code which failed to render."
        )
        
        user_content = (
            f"The following Mermaid code failed to render.\n"
            f"Error Message: {error_message}\n\n"
            f"Code:\n```mermaid\n{original_code}\n```\n\n"
            "FIX IT. Return ONLY the corrected Mermaid code. No explanations."
        )
        
        response = self.client.chat.completions.create(
            model=self.text_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0 # Strict for fixing
        )
        
        content = response.choices[0].message.content
        usage = response.usage
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        } if usage else {}

        # Strip markdown if present
        if "```mermaid" in content:
            content = content.split("```mermaid")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return content, usage_dict
