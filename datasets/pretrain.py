from pytorch_pretrained_vit import ViT
model = ViT('B_16_imagenet1k', pretrained=True)

for name, module in model.named_modules():
    print(name)