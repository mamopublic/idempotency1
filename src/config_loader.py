import yaml
import os
from dotenv import load_dotenv

load_dotenv()

def load_config(config_path="config/config.yaml"):
    """Loads the main configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_prompts(prompts_path="config/prompts.yaml"):
    """Loads the system prompts."""
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    
    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)
