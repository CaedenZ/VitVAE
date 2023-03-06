from torchvision import transforms
import torch
from PIL import Image
from torch import nn


# Load ViT
from pytorch_pretrained_vit import ViT
model = ViT('B_16_imagenet1k', pretrained=True)
model.eval()

# Load image
# NOTE: Assumes an image `img.jpg` exists in the current directory
img = transforms.Compose([
    transforms.Resize((384, 384)), 
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])(Image.open('test.png')).unsqueeze(0)
print(img.shape) # torch.Size([1, 3, 384, 384])



model.fc = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True),
            nn.Unflatten(dim=1,unflattened_size=(32, 3, 3)),
            nn.ConvTranspose2d(32, 16, 6,stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 6, stride=2,padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 5, stride=2,padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.ConvTranspose2d(3, 3, 3, stride=2,padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.ConvTranspose2d(3, 3, 3, stride=2,padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.ConvTranspose2d(3, 3, 3, stride=2,padding=1, output_padding=1), 
        )


# Classify
with torch.no_grad():
    outputs = model(img)
print(outputs.shape)  # (1, 1000)