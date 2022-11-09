import os
import numpy as np

from ffcv.loader import Loader, OrderOption
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, IntField, RGBImageField
from ffcv.fields.decoders import NDArrayDecoder, IntDecoder, SimpleRGBImageDecoder
from ffcv.transforms import ToTensor, ToDevice

def convert_to_ffcv(dataset, ffcv_filepath):
	print (f"Creating ffcv format {ffcv_filepath}")
	writer = DatasetWriter(ffcv_filepath,
				{'images' : RGBImageField(max_resolution=224),
				'labels' : IntField(),
				'embeddings' : NDArrayField(shape=(1024,), dtype=np.dtype('float32'))})
	writer.from_indexed_dataset(dataset)

def get_dataloaders(debug, root_dir, train_shuffle, test_shuffle, batch_size, num_workers, train_ds=None, test_ds=None):

	if debug:
		train_filepath = os.path.join(root_dir, f"temp_train_{len(train_ds)}.beton")
		test_filepath = os.path.join(root_dir, f"temp_test_{len(test_ds)}.beton")
	else:
		train_filepath = os.path.join(root_dir, "train.beton")
		test_filepath = os.path.join(root_dir, "test.beton")

	if not os.path.exists(train_filepath):
		convert_to_ffcv(train_ds, train_filepath)

	if not os.path.exists(test_filepath):
		convert_to_ffcv(test_ds, test_filepath)

	pipelines = {'images' : [SimpleRGBImageDecoder()], 
				 "labels" : [IntDecoder(), ToTensor(), ToDevice("cuda")], 
				 "embeddings" : [NDArrayDecoder(), ToTensor(), ToDevice("cuda")]}

	train_shuffle = OrderOption.RANDOM if train_shuffle else OrderOption.SEQUENTIAL
	test_shuffle = OrderOption.RANDOM if test_shuffle else OrderOption.SEQUENTIAL

	train_dl = Loader(train_filepath, batch_size=batch_size, 
					  num_workers=num_workers, order=train_shuffle, pipelines=pipelines)

	test_dl = Loader(test_filepath, batch_size=batch_size, 
					  num_workers=num_workers, order=test_shuffle, pipelines=pipelines)

	return train_dl, test_dl

	