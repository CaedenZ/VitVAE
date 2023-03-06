
import torch
from pacmandata import PacmanDataset

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

pacmandata = PacmanDataset(csv_file='blackened.csv',root_dir='blackened')
dataloader = DataLoader(pacmandata,batch_size=32, shuffle=False)

print(next(iter(dataloader)).shape)

cols, rows = 3, 3
figure = plt.figure(figsize=(8, 8))
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(pacmandata), size=(1,)).item()
    img = pacmandata[sample_idx]
    figure.add_subplot(rows, cols, i)
    # plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()