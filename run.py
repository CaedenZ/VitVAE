import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from vision_transformer import Trainer, ViT, ViTData
from vision_transformer.utils import Cfg
from pytorch_pretrained_vit import ViT


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training"
    )

    return parser.parse_args()


def main():
    
    args = make_parser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    config_file = args.config_file
    config_file_dir = str(Path(config_file).resolve().parent)
    config = Cfg(config_file)
    dataset_params = config.get_params(type="dataset")
    dataloader_params = config.get_params(type="dataloader")
    model_params = config.get_params(type="model")
    learning_params = config.get_params(type="learning")

    data = ViTData(name=dataset_params["name"], save_dir=dataset_params["dir"])
    train_loader, valid_loader = data.create_dataloader(
        batch_size=dataloader_params["batch_size"], shuffle=dataloader_params["shuffle"]
    )

    # model = ViT(
    #     in_channels=dataset_params["in_channels"],
    #     num_classes=dataset_params["num_classes"],
    #     embed_dim=model_params["embed_dim"],
    #     patch_size=model_params["patch_size"],
    #     image_size_h=dataset_params["image_size_h"],
    #     image_size_w=dataset_params["image_size_w"],
    #     num_blocks=model_params["num_blocks"],
    #     nb_head=model_params["heads"],
    #     hidden_dim=model_params["hidden_dim"],
    #     dropout=model_params["dropout"],
    #     latent_dim=model_params["latent_dim"]
    # )
    # model.to(device)
    
    model = ViT('B_16_imagenet1k', pretrained=True,image_size = 64)
    model.eval()
    model.fc = nn.Sequential(
            nn.Linear(768, 32 * 32 * 3),
            nn.ReLU(True),
            nn.Linear(32 * 32 * 3, 3 * 64 * 64),
            nn.ReLU(True),
            nn.Unflatten(dim=1,unflattened_size=(3, 64, 64)),

            # nn.ConvTranspose2d(32, 16, 10,stride=2, output_padding=0),
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(16, 8, 7, stride=2,padding=1, output_padding=1),
            # nn.BatchNorm2d(8),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(8, 3, 3, stride=2,padding=1, output_padding=1)

            # nn.ConvTranspose2d(32, 16, 6,stride=2, output_padding=0),
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(16, 8, 6, stride=2,padding=1, output_padding=1),
            # nn.BatchNorm2d(8),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(8, 3, 5, stride=2,padding=1, output_padding=1),
            # nn.BatchNorm2d(3),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(3, 3, 3, stride=2,padding=1, output_padding=1),
            # nn.BatchNorm2d(3),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(3, 3, 3, stride=2,padding=1, output_padding=1),
            # nn.BatchNorm2d(3),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(3, 3, 3, stride=2,padding=1, output_padding=1), 
        )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=learning_params["learning_rate"], momentum=0.9
    )
    trainer = Trainer(
        epochs=learning_params["epochs"],
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=config_file_dir,
        image_dir='result'
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
