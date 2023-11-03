import numpy as np
from PIL import Image
from torchvision import transforms
import os

# Define the directory where your test images are stored
image_dir = "C:\\Users\\donir\\models\\yolov5\\test"
binary_dir = "C:\\Users\\donir\\models\\yolov5\\binary_data"
os.makedirs(binary_dir, exist_ok=True)

# Define the transformations required for YOLOv5 input
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to the input size expected by YOLOv5
    transforms.ToTensor(),  # Convert to tensor
    # Normalize using standard ImageNet means and standard deviations
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Loop through each image in the directory
for i, filename in enumerate(os.listdir(image_dir)):
    if filename.lower().endswith((".jpg", ".png")):
        # Open the image
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        
        # Apply the transformations
        image_tensor = transform(image)
        
        # Save the tensor as a binary file
        binary_path = os.path.join(binary_dir, f"data{i}.txt")
        with open(binary_path, "wb") as binary_file:
            binary_file.write(image_tensor.numpy().tobytes())
            
        # Optionally, save the tensor as an image file for verification
        image.save(os.path.join(binary_dir, f"data{i}.jpg"))
        
        print(f"Processed and saved {filename} as binary and image.")
