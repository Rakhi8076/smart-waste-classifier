import torch
import torch.nn as nn
from torchvision import models

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
model.load_state_dict(torch.load("model/best_model_ft.pth", map_location="cpu"))
model.eval()
print("Model loaded OK")