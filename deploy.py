import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN Model
class BrainTumorDetector(nn.Module):
    def __init__(self):
        super(BrainTumorDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.pool(out)
        out = self.relu(self.conv2(out))
        out = self.pool(out)
        out = out.view(-1, 64 * 56 * 56)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Create an instance of the model class
model = BrainTumorDetector()

# Load the state dictionary (assuming it is saved as 'model_state_dict.pth')
state_dict = torch.load('./brain_tumor_detector.pt', map_location=torch.device('cpu'))

# Update the model with the state dictionary
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit webpage
st.title('Brain Tumor Detection')

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an RGB image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Predict button
    if st.button('Predict'):
        # Convert the image to a tensor
        image = transform(image).unsqueeze(0)

        # Get predictions from the model
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # Display results
        if predicted.item() == 0:
            st.write("This Image does NOT contain a brain tumor.")
        else:
            st.write("This Image contains a brain tumor.")
