import hydra
import torch
import wandb
import time

from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from utils import get_dataloaders
from dataset import load_datasets
from datasets import load_dataset

def get_class_text_embeddings(classes, device):
	texts = [f"A photo of {class_name}" for class_name in classes]

	model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
	processor = CLIPProcessor.from_pretrained(model_name)
	model = CLIPModel.from_pretrained(model_name).to(device)

	with torch.no_grad():
		text_embeddings = torch.zeros([1000, 1024]).to(device)
		for idx, text in enumerate(texts):
			inputs = processor(text=text, return_tensors="pt")
			inputs['input_ids'] = inputs['input_ids'].to(device)
			inputs['attention_mask'] = inputs['attention_mask'].to(device)

			text_embeddings[idx] = model.get_text_features(**inputs)[0]

		return text_embeddings

@hydra.main(config_path="confs", config_name="config")
def main(cfg):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dataset = load_dataset("imagenet-1k")
	classes = dataset["train"].features["label"].names

	class_embeddings = get_class_text_embeddings(classes, device) # 1000 x 1024

	datasets_dict = load_datasets(cfg.dataset.root_dir, cfg.dataset.debug)
	train_ds = datasets_dict['train']
	test_ds = datasets_dict['validation']
	_, test_dl = get_dataloaders(cfg.dataset, train_ds, test_ds)

	total_correct = 0
	total_iters = 0

	with wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags
    ) as wandb_run:

		start_time = time.time()

		for data in tqdm(test_dl):
			# Put embeddings on device
			image_embeddings = data['embeddings'].to(device) # Batch_size x 1024
			labels = data['labels'].to(device)

			# Compute dot product - Batch_size x 1000
			dot_prod = torch.matmul(image_embeddings, class_embeddings.T)

			# Argmax to get the predicted class - Batch_size 
			preds = dot_prod.argmax(dim=1)

			# Compute accuracy
			correct = (preds == labels).sum() / labels.shape[0]
			total_correct += correct

			total_iters += 1

		total_testing_time = time.time() - start_time
		wandb_run.log({"accuracy" : total_correct / total_iters, "testing_time_in_mins" : total_testing_time})

		print (f"Accuracy: {total_correct / total_iters}")
		print (f"--------------------------Testing time : {total_testing_time} mins--------------------------")

if __name__ == "__main__":
	main()