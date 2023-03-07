from pytorch_pretrained_vit import ViT
from torch import nn

model = ViT('B_16_imagenet1k', pretrained=True,image_size = 64)
model.eval()
# model.fc = nn.Sequential(
#         nn.Linear(768, 128),
#         nn.ReLU(True),
#         nn.Linear(128, 3 * 3 * 32),
#         nn.ReLU(True),
#         nn.Unflatten(dim=1,unflattened_size=(32, 3, 3)),

#         nn.ConvTranspose2d(32, 16, 10,stride=2, output_padding=0),
#         nn.BatchNorm2d(16),
#         nn.ReLU(True),
#         nn.ConvTranspose2d(16, 8, 7, stride=2,padding=1, output_padding=1),
#         nn.BatchNorm2d(8),
#         nn.ReLU(True),
#         nn.ConvTranspose2d(8, 3, 3, stride=2,padding=1, output_padding=1)
#     )

for name, module in model.named_modules():
    print(name)