from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def get_dataloaders(cfg, train_ds, test_ds):
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                          shuffle=cfg.train_shuffle, num_workers=cfg.num_workers,
                          pin_memory=cfg.pin_memory)

    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size,
                         shuffle=cfg.test_shuffle, num_workers=cfg.num_workers,
                         pin_memory=cfg.pin_memory)

    return train_dl, test_dl
