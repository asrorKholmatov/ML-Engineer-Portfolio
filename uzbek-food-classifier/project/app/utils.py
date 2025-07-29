from torchvision.models import EfficientNet_B0_Weights
import json

def get_transform():
    weights = EfficientNet_B0_Weights.DEFAULT
    return weights.transforms()

def load_class_names(path: str):
    with open(path, 'r') as f:
        return json.load(f)