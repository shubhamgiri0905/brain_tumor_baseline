import json
import os
from datetime import datetime

REGISTRY_PATH = "registry.json"

def save_model_version(accuracy, model_path, notes=""):
    # Load existing registry or create a new one
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = []

    # Determine next version number
    version = len(registry) + 1

    # Build metadata
    metadata = {
        "version": version,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": round(accuracy, 4),
        "model_path": model_path,
        "notes": notes
    }

    # Append to registry
    registry.append(metadata)

    # Save updated registry
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)

    print(f"✅ Model version v{version} saved to registry.json")
