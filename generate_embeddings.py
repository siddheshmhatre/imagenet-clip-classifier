import time
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
from transformers import CLIPModel, CLIPFeatureExtractor

def main():
	# Hard-coding args for now
	model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
	batch_size = 384
	num_workers = 16
	dataset_name = "imagenet-1k"

	# Create data loaders
	dataset = load_dataset(dataset_name)
	dataset.set_format('torch')

	feat_ext = CLIPFeatureExtractor()
	def transform(example):
		example['image'] = feat_ext(example['image'], return_tensors="pt").data['pixel_values']
		return example

	dataset.set_transform(transform) # Doing this stops returning "image" as a tensor
	train_ds, val_ds = dataset['train'], dataset['validation']

	train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)

	val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

	# Create model
	model = CLIPModel.from_pretrained(model_name)
	model.to('cuda')
	print (f"MODEL {model.device}")

	embeddings_dict = {}
	embeddings_dict['embeddings'] = None
	embeddings_dict['labels'] = None
	embeddings_dict['images'] = None

	# Iterate through dataset
	for data in tqdm(train_dl):

		with torch.no_grad():
			# Generate embeddings 
			data['image'] = data['image'].to('cuda')

			embeddings = model.get_image_features(data['image'])

			# Store embeddings
			if embeddings_dict['embeddings'] == None:
				embeddings_dict['embeddings'] = embeddings
				embeddings_dict['labels'] = data['label']
				embeddings_dict['images'] = data['image']
			else:
				embeddings_dict['embeddings'] = torch.vstack([embeddings_dict['embeddings'], embeddings])
				embeddings_dict['labels'] = torch.vstack([embeddings_dict['labels'], data['label']])
				embeddings_dict['images'] = torch.vstack([embeddings_dict['images'], data['image']])

	# Save to disk
	embeddings_dict['embeddings'] = embeddings_dict['embeddings'].to('cpu')
	embeddings_dict['images'] = embeddings_dict['images'].to('cpu')
	torch.save(embeddings_dict, "imagenet1k_train_clip_embeddings.pt")

if __name__ == "__main__":
	main()
