import os
import wandb
import torch
from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.optim import SGD, Adam
import dataset_ffcv as ffcv
from dataset import load_datasets
from utils import get_dataloaders
from model import MLP

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

def train(model, train_dl, test_dl, optimizer, epoch, test_fn, wandb_run, cfg): 
    start_time = time.time()

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

        total_iters += 1

        total_loss += loss
        if index % cfg.logging_freq == 0:
            print(f"Epoch {epoch}, Iteration {total_iters} / {len(train_dl)}, Batch loss {loss}, Average Total loss {total_loss / (total_iters)}")

        wandb_run.log({"batch_train_loss" : loss})

    wandb_run.log({"train_loss" : total_loss / total_iters, "epoch": epoch})

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

    if cfg.test:
        test_fn(model, test_dl, epoch, wandb_run)

@hydra.main(config_path="confs", config_name="config")
def main(cfg):

    torch.manual_seed(cfg.seed)

    with wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    ) as wandb_run:

        datasets_dict = load_datasets(cfg.dataset.root_dir, cfg)
        train_ds = datasets_dict['train']
        test_ds = datasets_dict['validation']

        if cfg.dataset.type == "pytorch":
            train_dl, test_dl = get_dataloaders(cfg.dataset, train_ds, test_ds)
        elif cfg.dataset.type == "ffcv":
            train_dl, test_dl = ffcv.get_dataloaders(cfg.dataset, train_ds, test_ds)

        # Initialize model and perform logging
        model = MLP(**cfg.model).to('cuda')

        if cfg.wandb.watch:
            wandb_run.watch(model, log='all', log_freq=cfg.logging_freq, log_graph=True)
        
        # Create optimizer
        optim = get_optimizer(cfg, model)

        # Call training loop
        for epoch in range(cfg.num_epochs):
            train(model, train_dl, test_dl, optim, epoch, test, wandb_run, cfg)

if __name__ == "__main__":
    main()