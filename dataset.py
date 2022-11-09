import os
import numpy as np
from torch.utils.data import Dataset

class ImageNet1kEmbeddings(Dataset):
	def __init__(self, root_dir, debug) -> None:
		super().__init__()

		self.root_dir = root_dir
		self.files = list(os.listdir(self.root_dir))
		self.files = sorted(self.files, key=lambda x: int(x.split('.')[0]))
		self.debug = debug

	def __getitem__(self, index) -> dict:
		filepath = os.path.join(self.root_dir, self.files[index])
		data = np.load(filepath)
		images = np.transpose(data['images'], (0, 2, 3, 1)).astype(np.uint8)
		return (images[0], data['labels'][0], data['embeddings'][0])

	def __len__(self) -> int:
		if self.debug:
			return min(len(self.files), 81920)
		else:
			return len(self.files)

def load_datasets(root_dir, debug) -> dict:
	datasets_dict = {}
	datasets_dict['train'] = ImageNet1kEmbeddings(os.path.join(root_dir, 'train'), debug)
	datasets_dict['validation'] = ImageNet1kEmbeddings(os.path.join(root_dir, 'validation'), debug)
	datasets_dict['test'] = ImageNet1kEmbeddings(os.path.join(root_dir, 'test'), debug)
	return datasets_dict