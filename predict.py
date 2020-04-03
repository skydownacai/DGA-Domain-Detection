from Model import ClassfierUsingLstm
from loguru import logger
from Utils import DataLoader
from Config import Param
import torch
from DataStructure import Example

batch_predict_size = 5 # 批量检测个数

domains_need_predict = [
	"baidu.com","01ol.ee4kdushuba.com"," nhy655.3322.org","tmall.com","office.com","taobao.com","dropbox.com"
]# 需要预测的域名放在一个列表里面,如果要通过其他方式读取，请保持变量名不变

#加载字符表
vocab  = torch.load(Param.data_dir + "vocab.pt")

model = ClassfierUsingLstm(
	vocab_size=vocab.size,
	char_embedding_dim = 128,
	hidden_size=100,
	num_layers=2,
	bidirectional = True,
	dense_size = 100 #初始化模型为你需要预测时候的参数
).from_pretrained(Param.cached_dir + "pytorch_model_12.bin")  #通过文件加载模型的具体参数 从而恢复模型
model.to(torch.device(Param.device))
with torch.no_grad():
	ptr = 0
	while ptr < len(domains_need_predict):
		batch_domains = domains_need_predict[ptr : ptr + batch_predict_size]
		batch_max_len = max(list(map(len,batch_domains)))

		examples = []
		for domain in batch_domains:
			#把每个字符转化为此表里面的id
			char_ids = []
			for char in domain:
				char_ids.append(vocab[char])
			if len(char_ids) <= batch_max_len:
				char_ids = char_ids + [0] * batch_max_len
			char_ids = char_ids[:batch_max_len]
			assert len(char_ids) == batch_max_len
			examples.append(
				Example(
					domain_len = len(domain),
					domain_name = domain,
					char_ids = char_ids,
					label = None
				)
			)

		batch = DataLoader.examples_2_Batch_Input_Feature(
			batch_examples = examples
		)
		#char_ids的shape为 1 x domain_len
		prob = model(batch)
		labels = prob.argmax(dim = 1)
		for idx in range(len(batch_domains)):
			domain = batch_domains[idx]
			label = labels[idx].item()
			if label == 1 : label = "恶意地址"
			if label == 0 : label = "非恶意地址"
			logger.info("%s 检测结果为 : %s" % (domain,label))
		ptr += batch_predict_size