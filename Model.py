import torch
from DataStructure import BatchInputFeature
from typing import Dict
from Config import Param
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Utils import init_logger

logger = init_logger("train",Param.log_dir)

DEVICE = torch.device(Param.device)

class ClassfierUsingLstm(torch.nn.Module):

	def __init__(self,
				 vocab_size : int ,
				 char_embedding_dim : int,
				 hidden_size : int,
				 num_layers : int,
				 bidirectional : bool,
				 dense_size : int
				 ):
		super().__init__()
		self.char_embedding = torch.nn.Embedding(
			vocab_size + 5 ,char_embedding_dim,padding_idx = 0
		)
		self.bidirectional = bidirectional
		self.encoder = torch.nn.LSTM(
			input_size = char_embedding_dim,
			hidden_size = hidden_size,
			num_layers = num_layers,
		    bidirectional = bidirectional)

		encoder_dim = (bidirectional + 1) * hidden_size
		self.dense = torch.nn.Sequential(
			torch.nn.Linear(encoder_dim,dense_size),
			torch.nn.Tanh(),
			torch.nn.Linear(dense_size,2),
			torch.nn.Softmax(dim = -1)
		)
		self.loss = torch.nn.CrossEntropyLoss(
			weight = torch.tensor([1.0 , 5],dtype = torch.float)
		)
	def forward(self, batch : BatchInputFeature ) -> torch.tensor :

		char_embedding = self.char_embedding(batch.char_ids.to(DEVICE))
		char_embedding_packed = pack_padded_sequence(
			input = char_embedding,
			batch_first = True,
			lengths = batch.domain_lens,
			enforce_sorted = False
		)
		_,(hn,cn) = self.encoder(char_embedding_packed)
		if not self.bidirectional:
			batch_domain_encoded = hn[-1]
		else:
			batch_domain_encoded = torch.cat(
				[hn[-1],hn[-2]],dim = 1
			)
		prob = self.dense(batch_domain_encoded)
		return prob

	def loss_fn(self,prob : torch.tensor ,label : torch.tensor ):
		return self.loss(prob,label).cpu()

	def from_pretrained(self,cached_path_dir, mode = "eval"):
		logger.info("load pre_trained model from %s " % cached_path_dir)
		model_state_dict = torch.load(cached_path_dir, map_location=DEVICE)
		self.load_state_dict(model_state_dict)
		if mode == "eval":
			self.eval()
		if mode == "train":
			self.train()
		return self
