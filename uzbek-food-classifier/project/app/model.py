import torch
import torch.nn as nn
import torchvision.models as models

def load_model(weights_path: str,
               num_classes: int,
               device: torch.device):
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model