import torch.nn as nn
import torch.nn.functional as F

def get_norm_layer(norm_layer):
	if norm_layer == "layer":
		return nn.LayerNorm
	elif norm_layer == "batch":
		return nn.BatchNorm1d
	return None

def get_activation(activation):
    if activation == "relu":
        return nn.ReLU

class MLP(nn.Module):
	def __init__(self, input_dim=1024, 
				 num_classes=1000, layers=[1024], 
				 activation="relu", norm_layer="batch") -> None:
		super().__init__()

		self.activation = get_activation(activation)
		self.norm_layer = get_norm_layer(norm_layer)
		self.layers = nn.ModuleList([nn.Linear(input_dim, layers[0])])

		if self.norm_layer is not None:
			self.layers.append(self.norm_layer(layers[0]))

		if len(layers) > 1:
			for i in range(1, len(layers)):
				self.layers.append(nn.Linear(layers[i-1], layers[i]))

				if self.norm_layer is not None:
					self.layers.append(self.norm_layer(layers[i]))

				self.layers.append(self.activation())

		self.layers.append(nn.Linear(layers[-1], num_classes, bias=False))

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x