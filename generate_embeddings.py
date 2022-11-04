import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
from transformers import CLIPModel, CLIPFeatureExtractor

def normalize_embeddings(embeddings):
	l2 = np.atleast_1d(np.linalg.norm(embeddings, ord=2, axis=-1))
	l2[l2 == 0] = 1
	return embeddings / np.expand_dims(l2, axis=-1)

def main():
	# Hard-coding args for now
	model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
	batch_size = 1
	num_workers = 4
	dataset_name = "imagenet-1k"
	embeddings_dir = "imagenet1k_clip_embeddings"

	# Create data loaders
	dataset = load_dataset(dataset_name)
	dataset.set_format('torch')

	feat_ext = CLIPFeatureExtractor()
	def transform(example):
		example['image'] = feat_ext(example['image'], return_tensors="pt").data['pixel_values']
		return example

	dataset.set_transform(transform) # Doing this stops returning "image" as a tensor

	# Create model
	model = CLIPModel.from_pretrained(model_name)
	model.to('cuda')
	print (f"MODEL {model.device}")

	counter = 1

	# Iterate through dataset
	for dataset_type in dataset.keys():
		print (f"Processing {dataset_type}")
		dl = DataLoader(dataset[dataset_type], batch_size=batch_size, num_workers=num_workers, shuffle=False)

		embeddings_dir = os.path.join(embeddings_dir, dataset_type)
		if not os.path.exists(embeddings_dir):
			os.mkdir(embeddings_dir)

		for data in tqdm(dl):

			with torch.no_grad():
				# Generate embeddings 
				data['image'] = data['image'].to('cuda')

				embeddings = model.get_image_features(data['image']).to('cpu').numpy()
				data['image'] = data['image'].to('cpu').numpy()
				data['label'] = data['label'].to('cpu').numpy()

				embeddings = normalize_embeddings(embeddings)

				filepath = f"{embeddings_dir}/{counter}"
				np.savez(filepath, embeddings=embeddings, images=data['image'], labels=data['label'])

				counter += 1

if __name__ == "__main__":
	main()
