import numpy as np
import torch
import sys
from models.experimental import attempt_load

assert len(sys.argv) == 2
model = torch.load('store/yolov5s_model.pt')

with open(sys.argv[1], "rb") as f:
    data = f.read()
x_test = np.frombuffer(data, dtype=np.float32)

x_test_tensor = torch.from_numpy(x_test).reshape([-1, 3, 640, 640])

with torch.no_grad():
    predictions = model(x_test_tensor)

print(predictions)
