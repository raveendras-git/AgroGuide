import torch
import torch.nn as nn
from torchvision import models

# Define the class names as a list
classnames = [
    'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight',
    'Potato healthy', 'Potato Late blight', 'Tomato Target Spot', 
    'Tomato Tomato mosaic virus', 'Tomato Yellow Leaf Curl Virus',
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato healthy',
    'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 
    'Tomato Spider mites'
]

def get_classname(index):
    return classnames[index]

def load_model(model_path):
    # Load the ResNet18 model pretrained on ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Updated to use the new 'weights' argument
    
    # Modify the final layer to match the number of disease classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classnames))  # Adjust based on your dataset

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return model
