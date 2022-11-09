import torch.nn as nn
import torch.nn.functional as F

def get_activation(activation):
    if activation == "relu":
        return nn.ReLU

class MLP(nn.Module):
	def __init__(self, input_dim=1024, 
				 num_classes=1000, layers=[1024], 
				 activation="relu", batch_norm=False) -> None:
		super().__init__()

		self.activation = get_activation(activation)
		self.batch_norm = batch_norm
		self.layers = nn.ModuleList([nn.Linear(input_dim, layers[0])])

		if batch_norm:
			self.layers.append(nn.BatchNorm1d(layers[0]))

		if len(layers) > 1:
			for i in range(1, len(layers)):
				self.layers.append(nn.Linear(layers[i-1], layers[i]))

				if self.batch_norm:
					self.layers.append(nn.BatchNorm1d(layers[i]))

				self.layers.append(self.activation())

		self.layers.append(nn.Linear(layers[-1], num_classes))

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x