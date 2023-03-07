from pytorch_pretrained_vit import ViT
from skimage import io
from PIL import Image, ImageDraw
from torchsummary import summary
import matplotlib.pyplot as plt
import copy
import numpy as np

# model = ViT('B_16_imagenet1k', pretrained=True,image_size = 64)
# model.eval()
# summary(model.cuda(), (3, 64, 64))
# model
# print(model.summary())
im = Image.open('test.png')
ims = [copy.copy(im),copy.copy(im),copy.copy(im),copy.copy(im)]
draw = ImageDraw.Draw(ims[0])
draw.rectangle([(0, 0), (im.size[0]/2, im.size[1]/2)], fill="black", outline=None, width=1)
draw = ImageDraw.Draw(ims[1])
draw.rectangle([(im.size[0]/2, 0), (im.size[0], im.size[1]/2)], fill="black", outline=None, width=1)
draw = ImageDraw.Draw(ims[2])
draw.rectangle([(0, im.size[1]/2), (im.size[0]/2, im.size[1])], fill="black", outline=None, width=1)
draw = ImageDraw.Draw(ims[3])
draw.rectangle([(im.size[0]/2, im.size[1]/2), (im.size[0], im.size[1])], fill="black", outline=None, width=1)

arr = np.array([np.asarray(ims[0]),np.asarray(ims[1]),np.asarray(ims[2]),np.asarray(ims[3])])

print(arr.shape)


cols, rows = 2, 2
figure = plt.figure(figsize=(8, 8))
for i in range(1, cols * rows + 1):
    # sample_idx = torch.randint(len(pacmandata), size=(1,)).item()
    img = ims[i-1]
    figure.add_subplot(rows, cols, i)
    # plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img, cmap="gray")
plt.show()



# pred = model(image[None, ...])
# print(out)
# print(model.named_modules()))
# for name, module in model.named_modules():
#     print(name)