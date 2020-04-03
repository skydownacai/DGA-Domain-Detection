from Config import Param
from DataStructure import Example
from tqdm import tqdm
import pandas as pd
from typing import List,Tuple
import torch
from random import shuffle
from Utils import Vocab, init_logger

logger = init_logger("DataPreprocess",Param.log_dir)

def Create_Examples_and_Vocab( * input_file_names : str) -> Tuple[List[Example],Vocab]:
	'''
	把b.csv 与 w.csv的数据转化为Example的实例并保存下来
	:return:
	'''
	examples = []
	vocab = Vocab()
	max_domain_len = 0
	for filename in input_file_names:
		df = pd.read_csv(open(filename,encoding="utf-8"))
		for idx in tqdm(range(len(df)),desc="转换文件%s中" % filename,total=len(df)):
			row = df.iloc[idx]
			domain_name = str(row[0]).replace(" ","").replace("\t","") #清洗
			is_black = not int(row[1])
			char_ids = []
			#给字典添加字符
			for char in domain_name:
				vocab += char


			for char in domain_name:
				char_idx = vocab[char]
				char_ids.append(char_idx)

			assert  len(char_ids) == len(domain_name)

			#补全
			if len(char_ids) <= Param.max_domain_len:
				char_ids = char_ids + [0] * Param.max_domain_len
			char_ids = char_ids[:Param.max_domain_len]

			assert len(char_ids) == Param.max_domain_len
			example = Example(
				domain_name = domain_name,
				label = is_black,
				char_ids = char_ids,
				domain_len = len(domain_name)
			)
			if len(domain_name) > max_domain_len:
				max_domain_len = len(domain_name)

			if idx == 0:
				logger.info('''
				Create Example:
					域名 : %s
					是否为恶意地址 : %d
					字符id : %s
					域名长度 : %d
				''' % (domain_name,is_black,"|".join(list(map(str,char_ids))),len(domain_name)))
			examples.append(example)

	logger.info("vocab size : %d " % vocab.size)
	logger.info("max domain len : %d " % max_domain_len)
	return examples,vocab

if __name__ == "__main__":

	examples,vocab = Create_Examples_and_Vocab(
		Param.data_dir + "b.csv",
		Param.data_dir + "w.csv")

	torch.save(vocab,Param.data_dir + "/vocab.pt")
	#分割获得训练集和测试集 按 4 : 1 比例分割
	shuffle(examples)
	train_num = int(len(examples) * 4  / 5)  # 训练集数据
	valid_num = len(examples) - train_num
	train_examples = examples[: train_num]
	valid_examples = examples[train_num:]
	torch.save(train_examples,Param.data_dir + "/train_examples.pt")
	torch.save(valid_examples,Param.data_dir + "/valid_examples.pt")

