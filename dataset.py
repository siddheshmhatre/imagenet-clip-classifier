import os
import numpy as np
from torch.utils.data import Dataset

class ImageNet1kEmbeddings(Dataset):
	def __init__(self, data_dir, cfg) -> None:
		super().__init__()

		self.data_dir = data_dir
		self.files = list(os.listdir(self.data_dir))
		self.files = sorted(self.files, key=lambda x: int(x.split('.')[0]))
		self.cfg = cfg

	def __getitem__(self, index) -> dict:
		filepath = os.path.join(self.data_dir, self.files[index])
		return {key : value[0] for key, value in np.load(filepath).items()}

	def __len__(self) -> int:
		if self.cfg.dataset.debug:
			return self.cfg.dataset.batch_size * self.cfg.dataset.num_workers * 2
		else:
			return len(self.files)

def load_datasets(data_dir, cfg) -> dict:
	datasets_dict = {}
	datasets_dict['train'] = ImageNet1kEmbeddings(os.path.join(data_dir, 'train'), cfg)
	datasets_dict['validation'] = ImageNet1kEmbeddings(os.path.join(data_dir, 'validation'), cfg)
	datasets_dict['test'] = ImageNet1kEmbeddings(os.path.join(data_dir, 'test'), cfg)
	return datasets_dict