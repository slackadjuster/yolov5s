import torch
from models.experimental import attempt_load

# Load the model (Make sure that 'store/yolov5s.pt' is the correct path to your model file)
model = attempt_load('store/yolov5s.pt')

# Save the state dict with a .pt extension
torch.save(model.state_dict(), 'store/yolov5s_weights.pt')

# Save the entire model with a .pt extension
torch.save(model, 'store/yolov5s_model.pt')
