import os
import torch
from torch.utils.data import Dataset

class ImageNet1kEmbeddings(Dataset):
	def __init__(self, data_dir, debug=False) -> None:
		super().__init__()

		self.data_dir = data_dir
		self.files = list(os.listdir(self.data_dir))
		self.files = sorted(self.files, key=lambda x: int(x.split('.')[0]))
		self.debug = debug

	def __getitem__(self, index) -> dict:
		filepath = os.path.join(self.data_dir, self.files[index])
		data_dict = torch.load(filepath)

		data_dict['embeddings'] = data_dict['embeddings'][0]
		data_dict['labels'] = data_dict['labels'][0]

		return data_dict

	def __len__(self) -> int:
		if self.debug:
			return 4096
		else:
			return len(self.files)

def load_datasets(data_dir, debug) -> dict:
	datasets_dict = {}
	datasets_dict['train'] = ImageNet1kEmbeddings(os.path.join(data_dir, 'train'), debug)
	datasets_dict['validation'] = ImageNet1kEmbeddings(os.path.join(data_dir, 'validation'), debug)
	datasets_dict['test'] = ImageNet1kEmbeddings(os.path.join(data_dir, 'test'), debug)
	return datasets_dict