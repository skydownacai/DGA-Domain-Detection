import torch
from Config import Param
from typing import Generator, Union, List
from random import shuffle
from DataStructure import BatchInputFeature, Example

import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
import matplotlib.pyplot as plt
import os
import json

def init_logger(logger_name,logging_path):
	if logger_name not in Logger.manager.loggerDict:
		logger = logging.getLogger(logger_name)
		logger.setLevel(logging.DEBUG)
		handler = TimedRotatingFileHandler(filename = os.path.join(logging_path,"%s.txt" % logger_name), when='D', backupCount=7)
		datefmt = '%Y-%m-%d %H:%M:%S'
		format_str = '[%(asctime)s]: %(name)s [line:%(lineno)s] %(levelname)s  %(message)s'
		formatter = logging.Formatter(format_str, datefmt)
		handler.setFormatter(formatter)
		handler.setLevel(logging.INFO)
		logger.addHandler(handler)
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
		console.setFormatter(formatter)
		logger.addHandler(console)

	logger = logging.getLogger(logger_name)
	return logger

def plot_history(history_file):

	assert history_file.endswith(".json")
	with open(history_file) as f:
		history = json.loads(f.read())
	plt.figure(figsize = (15,15))
	ax1 = plt.subplot(2,1,1)
	ax2 = plt.subplot(2,1,2)
	train_loss = history["train_loss"]
	dev_loss = history["dev_loss"]
	ax1.plot(
		list(range(1,len(train_loss) + 1)),train_loss,label = "train_loss"
	)
	ax1.plot(
		list(range(1,len(train_loss) + 1)),dev_loss,label = "dev_loss"
	)
	ax1.legend(loc = "upper left")
	ax1.set_xlabel("iter")
	ax1.set_ylabel("loss")
	ax1.set_title("iter vs loss")


	ax2.plot(
		list(range(1,len(train_loss) + 1)),history["train_precision"],label = "train_precision"
	)
	ax2.plot(
		list(range(1,len(train_loss) + 1)),history["train_recall"],label = "train_recall"
	)
	ax2.plot(
		list(range(1,len(train_loss) + 1)),history["train_acc"],label = "train_acc"
	)
	ax2.plot(
		list(range(1,len(train_loss) + 1)),history["dev_precision"],label = "dev_precision"
	)
	ax2.plot(
		list(range(1,len(train_loss) + 1)),history["dev_recall"],label = "dev_recall"
	)
	ax2.plot(
		list(range(1, len(train_loss) + 1)), history["dev_acc"], label="dev_acc"
	)

	ax2.legend(loc = "upper right")
	ax2.set_xlabel("iter")
	ax2.set_ylabel("metrics")
	ax2.set_title("iter vs metrics")
	plt.savefig(Param.log_dir + "history.png")
	plt.show()

logger = init_logger("train",Param.log_dir)

def conclusion(history_file):
	assert history_file.endswith(".json")
	with open(history_file) as f:
		history = json.loads(f.read())
	for key in history:
		print(key,"max :",max(history[key]),"min :",min(history[key]))

class Vocab:

	def __init__(self):
		self.size = 1
		self.idx2char = ["[PAD]","[UNK]"]
		self.char2idx = {"[PAD]" : 0 , "[UNK]":1}

	def __add__(self, char):
		if char not in self.char2idx:
			self.idx2char.append(char)
			self.char2idx[char] = len(self.idx2char) - 1
			self.size += 1
		return self

	def __getitem__(self, item) -> Union[None,int,str]:
		if isinstance(item,str):
			if item not in self.char2idx : return 0
			return self.char2idx[item]
		if isinstance(item,int):
			return self.idx2char[item]

class DataLoader:

	def __init__(self,train_examples_dir : str,
				 valid_examples_dir : str,
				 test_examples_dir : str = None,
				 vocab : Vocab = None):
		self.vocab = vocab
		self.train_examples_dir = train_examples_dir
		self.valid_examples_dir = valid_examples_dir
		self.test_examples_dir = test_examples_dir

	@staticmethod
	def examples_2_Batch_Input_Feature(batch_examples : List) -> BatchInputFeature:

		return BatchInputFeature(
			domain_names=[example.domain_name for example in batch_examples],
			labels=torch.tensor([example.label if example.label != None else 0 for example in batch_examples], dtype=torch.long),
			char_ids=torch.cat(
				[torch.tensor(example.char_ids, dtype=torch.long).unsqueeze(0)
				 for example in batch_examples],
			),
			domain_lens=torch.tensor([example.domain_len for example in batch_examples], dtype=torch.long),
		)

	def Get_Batch_Iter(self,dataset : str,batch_size : int, show_example : bool = False) -> Generator:
		assert  dataset in ["train","valid","test"]
		example_dir = {
			"train" : self.train_examples_dir,
			"valid" : self.valid_examples_dir,
			"test"  : self.test_examples_dir,
		}[dataset]
		examples = torch.load(example_dir)
		ptr = 0 # 索引指针
		while True:
			if ptr + batch_size >= len(examples):
				ptr = 0
				shuffle(examples)
			if show_example and ptr == 0 :
				example = examples[0]
				print('''
				Example:
					域名 : %s
					是否为恶意地址 : %d
					字符id : %s
				''' % (example.domain_name,example.label,"|".join(list(map(str,example.char_ids)))))
				show_example = False
			batch_examples = examples[ptr : ptr + batch_size]
			yield DataLoader.examples_2_Batch_Input_Feature(batch_examples)
			ptr += batch_size
