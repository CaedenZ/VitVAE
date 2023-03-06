from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from vision_transformer.utils import AverageMeter, get_logger
import torchvision.transforms as T
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: Any,
        optimizer: Any,
        device: Any,
        save_dir: str,
    ) -> None:
        self.epochs = epochs
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir

        self.logger = get_logger(str(Path(self.save_dir).joinpath("log.txt")))
        self.best_loss = float("inf")

    def fit(self, model: nn.Module) -> None:
        for epoch in range(self.epochs):
            model.train()
            losses = AverageMeter("train_loss")
            # print(next(iter(self.train_loader)).shape)

            with tqdm(self.train_loader, dynamic_ncols=True) as pbar:
                pbar.set_description(f"[Epoch {epoch + 1}/{self.epochs}]")

                for tr_data in pbar:
                    # print(tr_data.shape)
                    tr_X = tr_data[0].to(self.device)
                    # tr_y = tr_data[1].to(self.device)
                    tr_X = tr_X.float()
                    self.optimizer.zero_grad()
                    out = model(tr_X)
                    # CrossEntropy
                    loss = nn.MSELoss()
                    output = loss(tr_X, out)
                    # loss = self.criterion(out, tr_y)
                    output.backward()
                    self.optimizer.step()

                    losses.update(output.item())

                    pbar.set_postfix(loss=losses.value)

            self.logger.info(f"(train) epoch: {epoch} loss: {losses.avg}")
            self.evaluate(model, epoch)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, epoch: int) -> None:
        model.eval()
        losses = AverageMeter("valid_loss")

        for va_data in tqdm(self.valid_loader):
            va_X = va_data[0].to(self.device)
            # va_y = va_data[1].to(self.device)
            va_X = va_X.float()
            out = model(va_X)
            # loss = self.criterion(out, va_y)
            loss = nn.MSELoss()
            output = loss(out, va_X)
            # loss = self.criterion(out, tr_y)
            # output.backward()
            losses.update(output.item())

        self.logger.info(f"(valid) epoch: {epoch} loss: {losses.avg}")
        self.generate_save_img(va_X,out,epoch)

        if losses.avg <= self.best_loss:
            self.best_acc = losses.avg
            torch.save(model.state_dict(), Path(self.save_dir).joinpath("best.pth"))

    def generate_save_img(self,x,out,epoch):
        # x = np.transpose( x.cpu(), (0, 2, 3, 1))
        # out = np.transpose( out.cpu(), (0, 2, 3, 1))
        print(x.shape)
        print(out.shape)
        # e_img = model.encode(test_sample)
        # d_img = model.decode(e_img)
        # random_idx = random.randint(0,batch_size-1)
        # x = x.transpose(0,2,3,1).astype("uint8")
        # out = out.transpose(0,2,3,1).astype("uint8")
        self.tensor_to_img(x,'input')
        self.tensor_to_img(out,'output')
        # fig = plt.figure()
        # f,ax = plt.subplots(1,2,figsize=(10,6))
        # ax[0].imshow(x)
        # ax[0].axis('off')
        # ax[0].set_title('Original Image')
        # ax[1].imshow(out)
        # ax[1].axis('off')
        # ax[1].set_title('Reconstructed Image')

        # plt.savefig(f'sample/image_{epoch}.png')


    def tensor_to_img(self,img,label):
        img = img.detach().cpu()

        if img.dim()==4: # 4D tensor
            bz = img.shape[0]
            c = img.shape[1]
            if bz==1 and c==1:  # single grayscale image
                img=img.squeeze()
            elif bz==1 and c==3: # single RGB image
                img=img.squeeze()
                img=img.permute(1,2,0)
            elif bz==1 and c > 3: # multiple feature maps
                img = img[:,0:3,:,:]
                img = img.permute(0, 2, 3, 1)[:]
                print('warning: more than 3 channels! only channels 0,1,2 are preserved!')
            elif bz > 1 and c == 1:  # multiple grayscale images
                img=img.squeeze()
            elif bz > 1 and c == 3:  # multiple RGB images
                img = img.permute(0, 2, 3, 1)
            elif bz > 1 and c > 3:  # multiple feature maps
                img = img[:,0:3,:,:]
                img = img.permute(0, 2, 3, 1)[:]
                print('warning: more than 3 channels! only channels 0,1,2 are preserved!')
            else:
                raise Exception("unsupported type!  " + str(img.size()))
        elif img.dim()==3: # 3D tensor
            bz = 1
            c = img.shape[0]
            if c == 1:  # grayscale
                img=img.squeeze()
            elif c == 3:  # RGB
                img = img.permute(1, 2, 0)
            else:
                raise Exception("unsupported type!  " + str(img.size()))
        elif img.dim()==2:
            pass
        else:
            raise Exception("unsupported type!  "+str(img.size()))


        img = img.numpy()  # convert to numpy
        img = img.squeeze()
        print(img.shape)

        if bz ==1:
            plt.imshow(img, cmap='gray')
            # plt.colorbar()
            # plt.show()
        else:
            fig = plt.figure()
            cols,rows = 4,3
            for idx in range(0,bz):
                plt.subplot(int(bz**0.5),int(np.ceil(bz/int(bz**0.5))),int(idx+1))
                # fig.add_subplot(rows, cols, idx)
                plt.imshow(img[idx].astype('uint8'), cmap='gray')
            plt.savefig(f'sample/image_{idx}_{label}.png')

        
        return img