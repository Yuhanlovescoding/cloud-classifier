import torch
import os

from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision import transforms

from utils import create_logger, Cutout
from dataset import get_dataset
from model import get_model
from trainer import ClassifierTrainer

BATCH_SIZE = 128 
NUM_WORKERS = 1
LR = 3e-4

def main(num_epochs: int = 60, dataset_dir: str = 'data', model_save_dir: str = 'model'):

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = get_model(weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    logger = create_logger(
        filename=os.path.join(model_save_dir, 'log.log'),
        logger_prefix=__file__
    )
    
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2,
                               saturation=0.2,
                               hue=0.1),
        transforms.ToTensor(),
        normalizer,
        Cutout(20),
    ])

    val_test_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalizer,
    ])

    train_dataset, val_dataset, test_dataset = get_dataset(dataset_dir, train_transform, val_test_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=num_epochs,
    )

    trainer = ClassifierTrainer(
        model=model, criterion=criterion, optimizer=optimizer, device=device, 
        train_loader=train_loader, valid_loader=val_loader, test_loader=test_loader,
        scheduler=scheduler, epochs=num_epochs, logger=logger, model_save_dir=model_save_dir)

    trainer.train()
    trainer.validate()


if __name__ == '__main__':
    main()