"""
Configuration for the Knowledge Module: paths and YAML config loading.
"""

import yaml
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
CONFIG_FILE = ROOT_DIR / "config" / "models.yaml"
DATASET_DIR = ROOT_DIR / "db"

# Load configuration
def load_config():
    """Load configuration from YAML file."""
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_embedding_config():
    configs = load_config()
    return configs['models']['embedding']

def get_splitting_config():
    configs = load_config()
    return configs['models']['splitter']