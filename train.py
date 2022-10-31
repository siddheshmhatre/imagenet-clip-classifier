import os
import wandb
import torch
from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.optim import SGD
from dataset import load_datasets
from utils import get_dataloaders
from model import MLP

import hydra

def test(model, test_dl, epoch, wandb_run):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    total_iters = 0

    with torch.no_grad():

        for index, sample in enumerate(test_dl):
            # Forward pass
            embeddings = sample['embeddings'].to('cuda')
            labels = sample['labels'].to('cuda')
            output = model(embeddings)

            # Apply softmax to output?
            # Compute arg max 
            # Calculate loss and num correct
            loss = F.cross_entropy(output, labels)
            total_loss += loss

            _, argmax = output.max(dim=1)
            print (argmax)

            correct = (argmax == labels).sum() / labels.shape[0]
            total_correct += correct

            total_iters += 1

    print (f"Epoch {epoch}, Test Loss {total_loss / total_iters}, Accuracy: {total_correct / total_iters}")
    wandb_run.log({"test_loss": total_loss / total_iters, "accuracy" : total_correct / total_iters})

def train(model, train_dl, test_dl, optimizer, logging_freq, epoch, test_fn, wandb_run):
    total_loss = 0.0
    total_iters = 0

    # Put model in train mode
    model.train()

    print (f"--------------------------Trainin Epoch: {epoch}--------------------------")

    for index, sample in enumerate(train_dl):
        # Zero grad
        optimizer.zero_grad()

        # Forward pass
        embeddings = sample['embeddings'].to('cuda')
        labels = sample['labels'].to('cuda')
        output = model(embeddings)

        # Compute loss
        loss = F.cross_entropy(output, labels)

        # Compute gradients 
        loss.backward()

        # Step optimizer
        optimizer.step()

        total_iters += 1

        total_loss += loss
        if index % logging_freq:
            print(f"Epoch {epoch}, Iteration {total_iters}, Batch loss {loss}, Average Total loss {total_loss / (total_iters)}")

        wandb_run.log({"batch_train_loss" : loss})

    wandb_run.log({"train_loss" : total_loss / total_iters})

    experiment_dir = os.getcwd()

    # Save model to disk
    with open(os.path.join(experiment_dir, f'model_{epoch}.ckpt'), 'wb') as f:
        torch.save({'epoch' : epoch, 
                    'model_state_dict' : model.state_dict(), 
                    'optimizer_state_dict' : optimizer.state_dict()},
                    f)

    print (f"--------------------------Testing Epoch: {epoch}--------------------------")
    test_fn(model, test_dl, epoch, wandb_run)

@hydra.main(config_path="confs", config_name="config")
def main(cfg):

    torch.manual_seed(cfg.seed)

    with wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags
    ) as wandb_run:

        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        datasets_dict = load_datasets(cfg.dataset.root_dir)
        train_ds = datasets_dict['train']
        test_ds = datasets_dict['validation']
        train_dl, test_dl = get_dataloaders(cfg.dataset, train_ds, test_ds)

        # Initialize model and perform logging
        model = MLP().to('cuda')
        
        # Create optimizer
        optim = SGD(model.parameters(), lr=cfg.optim.learning_rate, momentum=cfg.optim.momentum)

        # Call training loop
        for epoch in range(cfg.num_epochs):
            train(model, train_dl, test_dl, optim, cfg.logging_freq, epoch, test, wandb_run)

if __name__ == "__main__":
    main()