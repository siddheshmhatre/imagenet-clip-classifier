from torch.utils.data import DataLoader

def get_dataloaders(train_ds, test_ds, debug, root_dir, train_shuffle, test_shuffle, batch_size, num_workers):
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=train_shuffle, num_workers=num_workers,
                          pin_memory=True)

    test_dl = DataLoader(test_ds, batch_size=batch_size,
                         shuffle=test_shuffle, num_workers=num_workers,
                         pin_memory=True)

    return train_dl, test_dl
