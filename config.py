import os
import tomli
from typing import Dict, Any

def fix_escaping(text: str) -> str:
    # When JSON is loaded, \\\\ becomes \\ automatically
    # So we don't need to do any transformation
    return text

def load_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), 'config.toml')
    with open(config_path, 'rb') as f:
        config = tomli.load(f)
        return config

# Load configuration
config = load_config()

# Export variables for backward compatibility
MODEL = os.getenv("MODEL", config['model']['path'])
MIN_IMAGE_SIZE = int(os.getenv("MIN_IMAGE_SIZE", config['model']['min_image_size']))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", config['model']['max_image_size']))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", str(config['model']['use_torch_compile'])).lower() == "true"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", str(config['model']['enable_cpu_offload'])).lower() == "true"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", config['model']['output_dir'])

DEFAULT_NEGATIVE_PROMPT = config['prompts']['default_negative']
DEFAULT_ASPECT_RATIO = config['prompts']['default_aspect_ratio']

examples = config['prompts']['examples']
sampler_list = config['samplers']['list']
aspect_ratios = config['aspect_ratios']['list']
style_list = config['styles']