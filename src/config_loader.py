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
