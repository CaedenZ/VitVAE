from torch.utils.data import Dataset
import torch
import pandas as pd
import os
from skimage import io, transform
import numpy as np
from PIL import Image, ImageDraw
import copy
import random

class FusionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = Image.open(img_name)
        y = io.imread(img_name)
        y = np.transpose( y, (2, 0, 1))
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image}

        if self.transform:
            image = self.transform(image)

        ims = [copy.copy(image),copy.copy(image),copy.copy(image),copy.copy(image),copy.copy(image),copy.copy(image),copy.copy(image),copy.copy(image)]
        draw = ImageDraw.Draw(ims[0])
        draw.rectangle([(0, 0), (image.size[0]/2, image.size[1]/2)], fill="black", outline=None, width=1)
        draw = ImageDraw.Draw(ims[1])
        draw.rectangle([(image.size[0]/2, 0), (image.size[0], image.size[1]/2)], fill="black", outline=None, width=1)
        draw = ImageDraw.Draw(ims[2])
        draw.rectangle([(0, image.size[1]/2), (image.size[0]/2, image.size[1])], fill="black", outline=None, width=1)
        draw = ImageDraw.Draw(ims[3])
        draw.rectangle([(image.size[0]/2, image.size[1]/2), (image.size[0], image.size[1])], fill="black", outline=None, width=1)
        draw = ImageDraw.Draw(ims[4])
        draw.rectangle([(0, 0), (image.size[0], image.size[1]/2)], fill="black", outline=None, width=1)
        draw = ImageDraw.Draw(ims[5])
        draw.rectangle([(0, 0), (image.size[0]/2, image.size[1])], fill="black", outline=None, width=1)
        draw = ImageDraw.Draw(ims[6])
        draw.rectangle([(0, image.size[1]/2), (image.size[0], image.size[1])], fill="black", outline=None, width=1)
        draw = ImageDraw.Draw(ims[7])
        draw.rectangle([(image.size[0]/2, 0), (image.size[0], image.size[1])], fill="black", outline=None, width=1)

        arr = np.array([np.asarray(ims[0]),np.asarray(ims[1]),np.asarray(ims[2]),np.asarray(ims[3]),np.asarray(ims[4]),np.asarray(ims[5]),np.asarray(ims[6]),np.asarray(ims[7])])
        
        random.shuffle(arr)
        for i in range(len(arr)):
            if random.random() < 0.1:
                arr[i] = torch.zeros(64,64,3)
        arr = arr.swapaxes(1,3).reshape((24,64,64))
        return arr, y