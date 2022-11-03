import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
from transformers import CLIPModel, CLIPFeatureExtractor

def main():
	# Hard-coding args for now
	model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
	batch_size = 1
	num_workers = 4
	dataset_name = "imagenet-1k"
	embeddings_dir = "imagenet1k_clip_embeddings"
	dataset_type = "test"

	# Create data loaders
	dataset = load_dataset(dataset_name)
	dataset.set_format('torch')

	feat_ext = CLIPFeatureExtractor()
	def transform(example):
		example['image'] = feat_ext(example['image'], return_tensors="pt").data['pixel_values']
		return example

	dataset.set_transform(transform) # Doing this stops returning "image" as a tensor

	dl = DataLoader(dataset[dataset_type], batch_size=batch_size, num_workers=num_workers, shuffle=True)

	# Create model
	model = CLIPModel.from_pretrained(model_name)
	model.to('cuda')
	print (f"MODEL {model.device}")

	counter = 1

	embeddings_dir = os.path.join(embeddings_dir, dataset_type)
	if not os.path.exists(embeddings_dir):
		os.mkdir(embeddings_dir)

	# Iterate through dataset
	for data in tqdm(dl):

		with torch.no_grad():
			# Generate embeddings 
			data['image'] = data['image'].to('cuda')

			embeddings = model.get_image_features(data['image'])

			embeddings_dict = {}
			embeddings_dict['embeddings'] = embeddings
			embeddings_dict['labels'] = data['label']
			embeddings_dict['images'] = data['image']

			# Save to disk
			embeddings_dict['embeddings'] = embeddings_dict['embeddings'].to('cpu')
			embeddings_dict['images'] = embeddings_dict['images'].to('cpu')
			torch.save(embeddings_dict, f"{embeddings_dir}/{counter}.pt")

			counter += 1

			del embeddings_dict

if __name__ == "__main__":
	main()
