import os
import wandb
import torch
from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.optim import SGD, Adam
import dataset_ffcv as ffcv
from dataset import load_datasets
from utils import get_dataloaders, add_key_value_pair
from model import MLP
from torch.optim.lr_scheduler import CosineAnnealingLR

import pytorch_warmup as warmup

import hydra
import time

def get_optimizer(cfg, model):
    optim_class = None
    if cfg.optim_name.lower() == "adam":
        optim_class = Adam
    # Default to SGD
    else:
        optim_class = SGD

    return optim_class(model.parameters(), **cfg.optim[cfg.optim_name])

def get_lr_scheduler(cfg, optim, num_iters_per_epoch):
    lr_scheduler_class = None
    if cfg.lr_scheduler_name.lower() == "cosine":
        return CosineAnnealingLR(optim, T_max=num_iters_per_epoch * cfg.lr_scheduler.cosine.num_epochs)
    return None

def test(model, test_dl, epoch, wandb_run):
    # TODO - replace accuracy calculation with a meter object
    start_time = time.time()
    model.eval()
    total_correct = 0
    total_loss = 0.0
    total_iters = 0

    with torch.no_grad():

        for index, sample in enumerate(test_dl):
            # Forward pass
            _, labels, embeddings = sample

            labels = labels.to('cuda').squeeze()
            embeddings = embeddings.to('cuda')

            output = model(embeddings)

            # Apply softmax to output?
            # Compute arg max 
            # Calculate loss and num correct
            loss = F.cross_entropy(output, labels)
            total_loss += loss

            _, argmax = output.max(dim=1)

            correct = (argmax == labels).sum() / labels.shape[0]
            total_correct += correct

            total_iters += 1

    print (f"Epoch {epoch}, Test Loss {total_loss / total_iters}, Accuracy: {total_correct / total_iters}")
    wandb_run.log({"test_loss": total_loss / total_iters, "accuracy" : total_correct / total_iters, "epoch": epoch})

    end_time = time.time()
    total_testing_time = (end_time - start_time) / 60
    wandb_run.log({"testing_time_in_mins" : total_testing_time})
    print (f"--------------------------Testing time : {total_testing_time} mins--------------------------")

def train(model, train_dl, test_dl, optimizer, epoch, test_fn, wandb_run, lr_scheduler, logging_freq, test): 
    start_time = time.time()

    total_correct = 0
    total_loss = 0.0
    total_iters = 0

    # Put model in train mode
    model.train()

    print (f"--------------------------Trainin Epoch: {epoch}--------------------------")

    for index, sample in enumerate(train_dl):
        # Zero grad
        optimizer.zero_grad()

        # Forward pass
        _, labels, embeddings = sample
        labels = labels.to('cuda').squeeze()
        embeddings = embeddings.to('cuda')

        output = model(embeddings)

        # Compute loss
        loss = F.cross_entropy(output, labels)

        # Compute gradients 
        loss.backward()

        # Step optimizer
        optimizer.step()

        # Get predictions
        _, argmax = output.max(dim=1)

        correct = (argmax == labels).sum() / labels.shape[0]
        total_correct += correct

        total_iters += 1

        total_loss += loss
        if index % logging_freq == 0:
            print(f"Epoch {epoch}, Iteration {total_iters} / {len(train_dl)}, Batch loss {loss}, Average Total loss {total_loss / (total_iters)}")

        if lr_scheduler is not None:
            # Step lr scheduler
            lr_scheduler.step()
            learning_rate = lr_scheduler.get_last_lr()[0]
            wandb_run.log({"batch_train_loss" : loss, "learning_rate" : torch.tensor(learning_rate)})
        else:
            wandb_run.log({"batch_train_loss" : loss})

    wandb_run.log({"train_loss" : total_loss / total_iters, "train_accuracy" : total_correct / total_iters, "epoch": epoch})

    experiment_dir = os.getcwd()

    # Save model to disk
    with open(os.path.join(experiment_dir, f'model_{epoch}.ckpt'), 'wb') as f:
        torch.save({'epoch' : epoch, 
                    'model_state_dict' : model.state_dict(), 
                    'optimizer_state_dict' : optimizer.state_dict()},
                    f)

    end_time = time.time()
    total_training_time = (end_time - start_time) / 60
    wandb_run.log({"training_time_in_mins" : total_training_time})
    print (f"--------------------------Training time : {total_training_time} mins--------------------------")
    print (f"--------------------------Testing Epoch: {epoch}--------------------------")

    if test:
        test_fn(model, test_dl, epoch, wandb_run)

@hydra.main(config_path="confs", config_name="config")
def main(cfg):

    torch.manual_seed(cfg.seed)
    add_key_value_pair(cfg, "output_dir", os.getcwd())

    with wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    ) as wandb_run:

        datasets_dict = load_datasets(**cfg.dataset)
        train_ds = datasets_dict['train']
        validation_ds = datasets_dict['validation']
        test_ds = datasets_dict['test']

        if cfg.dataset_type == "pytorch":
            train_dl, validation_dl, _ = get_dataloaders(train_ds=train_ds, validation_ds=validation_ds, test_ds=test_ds, 
                                                         **cfg.dataset, **cfg.dataloader)
        elif cfg.dataset_type == "ffcv":
            train_dl, validation_dl, _ = ffcv.get_dataloaders(train_ds=train_ds, validation_ds=validation_ds, test_ds=test_ds, 
                                                              **cfg.dataset, **cfg.dataloader)

        # Initialize model and perform logging
        model = MLP(**cfg.model).to('cuda')

        if cfg.wandb.watch:
            wandb_run.watch(model, log='all', log_freq=cfg.logging_freq, log_graph=True)
        
        # Create optimizer
        optim = get_optimizer(cfg, model)

        # LR scheduler
        lr_scheduler = get_lr_scheduler(cfg, optim, len(train_dl))

        # Call training loop
        for epoch in range(cfg.num_epochs):
            train(model, train_dl, validation_dl, optim, epoch, test, wandb_run, lr_scheduler, cfg.logging_freq, cfg.test)

if __name__ == "__main__":
    main()