import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.lin1 = nn.Linear(1024, 1024)
		self.lin2 = nn.Linear(1024, 1000)

	def forward(self, x):
		x = F.relu(self.lin1(x))
		x = self.lin2(x)
		return x