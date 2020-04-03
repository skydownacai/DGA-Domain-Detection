import os
from typing import Dict
class Param:
	#path configuration
	device : str = "cuda:0" # 控制模型运行时
	root : str  = os.getcwd()
	data_dir : str  = os.path.join(root,"data/")
	log_dir : str = os.path.join(root,"log/")
	cached_dir : str = os.path.join(root,"model_saved/")

	#Model
	char_embedding_dim: int = 128
	hidden_size: int = 100
	num_layers: int = 2
	bidirectional: bool = True
	dense_size : int = 100
	positive_sample_loss_weight : float = 5
	#Train
	learning_rate = 3e-4
	warmup_proportion = 0.05
	max_domain_len = 130

	train_epoch_num: int = 15
	train_batch_size: int = 2000
	train_batch_num_one_epoch: int = int(800000 / train_batch_size)

	valid_batch_size = 500
	valid_batch_num_one_epoch: int = int(200000 / valid_batch_size)
