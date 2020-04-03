from Utils import DataLoader, init_logger, plot_history, conclusion
from Config import Param
import torch
from typing import  Tuple, Generator
from Model import ClassfierUsingLstm
from tqdm import  tqdm
import torch.optim as optim
import json
import numpy as np
import os

logger = init_logger("train",Param.log_dir)
DEVICE = torch.device(Param.device)

def BinarySeqLabel(pred: torch.tensor, truth: torch.tensor) -> Tuple:

	TP = ((pred == 1) & (truth == 1)).cpu().sum().item()
	TN = ((pred == 0) & (truth == 0)).cpu().sum().item()
	FN = ((pred == 0) & (truth == 1)).cpu().sum().item()
	FP = ((pred == 1) & (truth == 0)).cpu().sum().item()

	p = 0 if TP == FP == 0 else TP / (TP + FP)
	r = 0 if TP == FN == 0 else TP / (TP + FN)
	F1 = 0 if p == r == 0 else 2 * r * p / (r + p)
	acc = (TP + TN) / (TP + TN + FP + FN)

	return (p, r, F1, acc)

def warmup_linear(step_ratio, warmup_ratio =0.002) -> float:
	if step_ratio < warmup_ratio:
		return step_ratio / warmup_ratio
	return 1.0 - step_ratio

def train(model : torch.nn.Module,
			train_epoch_num : int ,
			train_batch_num_one_epoch : int,
			train_batch_iter : Generator,
			train_batch_size: int,

			valid_batch_iter : Generator,
			valid_batch_num_one_epoch: int,
			valid_batch_size: int):

	model = model.to(DEVICE)
	logger.info("__________________start_training___________________")
	# ---------------------优化器-------------------------
	optimizer = optim.Adam(model.parameters(), lr = Param.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

	T_losses, T_p, T_r, T_f1, T_acc = [], [], [], [], []
	V_losses, V_p, V_r, V_f1, V_acc = [], [], [], [], []
	history = {
		"train_loss": T_losses,
		"train_precision": T_p,
		"train_recall": T_r,
		"train_f1": T_f1,
		"train_acc": T_acc,
		"dev_loss": V_losses,
		"dev_precision": V_p,
		"dev_recall": V_r,
		"dev_f1": V_f1,
		"dev_acc": V_acc
	}
	global_step = 0
	t_total = train_epoch_num * train_batch_num_one_epoch
	for epoch in range(train_epoch_num):
		train_acc_num = 0
		pbar = tqdm(total = train_batch_num_one_epoch)
		losses, p, r, f1, acc = [], [], [], [], []
		for batch_idx in range(train_batch_num_one_epoch):
			optimizer.zero_grad()
			batch = train_batch_iter.__next__()
			labels = batch.labels.to(DEVICE)
			prob = model(batch)
			predict = prob.argmax(dim = 1)
			loss = model.loss_fn(prob,labels)
			b_p, b_r, b_f1, b_acc = BinarySeqLabel(predict.cpu(),labels.cpu())
			p.append(b_p)
			r.append(b_r)
			f1.append(b_f1)
			acc.append(b_acc)

			acc_num = (predict == labels).sum().item()
			train_acc_num += acc_num

			losses.append(loss.item())
			loss.backward()

			#update learing rate
			lr_this_step = Param.learning_rate * warmup_linear(global_step / t_total, Param.warmup_proportion)
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr_this_step

			optimizer.step()
			optimizer.zero_grad()
			global_step += 1

			pbar.set_description(
				'Epoch: %2d (learing rate : %.6f )| LOSS: %2.3f | F1: %1.3f | ACC: %1.3f| PRECISION: %1.3f | RECALL: %1.3f' % (
				epoch,lr_this_step,loss.item(), b_f1, b_acc, b_p, b_r))
			pbar.update(1)
			optimizer.step()

		train_acc = train_acc_num / (train_batch_size * train_batch_num_one_epoch)
		losses, p, r, f1, acc = list(map(np.mean, [losses, p, r, f1, acc]))
		logger.info('Train Epoch: %2d  (learing rate : %.4f )| LOSS: %2.3f | F1: %1.3f | ACC: %1.3f | PRECISION: %1.3f | RECALL: %1.3f' %
					(epoch,lr_this_step, losses, f1, train_acc , p, r))

		T_p.append(p)
		T_r.append(r)
		T_f1.append(f1)
		T_acc.append(train_acc)
		T_losses.append(losses)

		pbar.clear()
		pbar.close()

		valid_acc_num = 0
		losses, p, r, f1, acc = [], [], [], [], []

		with torch.no_grad():
			for batch_idx in range(valid_batch_num_one_epoch):
				batch = valid_batch_iter.__next__()
				labels = batch.labels.to(DEVICE)
				prob = model(batch)
				predict = prob.argmax(dim=1)
				loss = model.loss_fn(prob, labels)
				b_p, b_r, b_f1, b_acc = BinarySeqLabel(predict.cpu(), labels.cpu())
				p.append(b_p)
				r.append(b_r)
				f1.append(b_f1)
				acc.append(b_acc)
				losses.append(loss.item())
				acc_num = (predict == labels).sum().item()
				valid_acc_num += acc_num

		valid_acc = valid_acc_num / (valid_batch_size * valid_batch_num_one_epoch)
		losses, p, r, f1, acc = list(map(np.mean, [losses, p, r, f1, acc]))
		logger.info('Valid Epoch: %2d | LOSS: %2.3f | F1: %1.3f | ACC: %1.3f | PRECISION: %1.3f | RECALL: %1.3f' %
					(epoch, losses, f1, valid_acc, p, r))
		V_acc.append(valid_acc)
		V_p.append(p)
		V_r.append(r)
		V_f1.append(f1)
		V_losses.append(losses)

		with open(Param.log_dir + "History.json", "w", encoding="utf-8") as f:
			f.write(json.dumps(history))

		if len(V_acc) >= 2:
			plot_history(Param.log_dir + "History.json")

		#当前模型验证集的F1值 大于 之前的F1值,那么保存当前模型的参数值
		if len(V_acc) == 1 or V_f1 >= max(V_f1[:-1]):
			model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
			output_model_file = os.path.join(Param.cached_dir, "pytorch_model_%s.bin" % epoch)
			torch.save(model_to_save.state_dict(), output_model_file)



vocab = torch.load(Param.data_dir + "vocab.pt")
dataloader = DataLoader(
	train_examples_dir = Param.data_dir + "train_examples.pt",
	valid_examples_dir = Param.data_dir + "valid_examples.pt",
	vocab = vocab
)
model = ClassfierUsingLstm(
	vocab_size = vocab.size,
	char_embedding_dim = Param.char_embedding_dim,
	hidden_size = Param.hidden_size,
	num_layers = Param.num_layers,
	bidirectional = Param.bidirectional,
	dense_size = Param.dense_size
)
train(model = model,
	train_epoch_num = Param.train_epoch_num,
	train_batch_num_one_epoch = Param.train_batch_num_one_epoch,
	train_batch_iter = dataloader.Get_Batch_Iter(dataset="train",batch_size= Param.train_batch_size,show_example=True),
	train_batch_size = Param.train_batch_size,
	valid_batch_iter = dataloader.Get_Batch_Iter(dataset="valid",batch_size= Param.valid_batch_size,show_example = False),
	valid_batch_num_one_epoch = Param.valid_batch_num_one_epoch,
	valid_batch_size = Param.valid_batch_size)
conclusion(Param.log_dir + "History.json")