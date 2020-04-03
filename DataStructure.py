from typing import NamedTuple, List, Optional
import torch.tensor as Tensor

class Example(NamedTuple):

	domain_name : str #域名
	label : Optional[bool]  #是否为恶意地址
	char_ids : List[int] #每个字符在vocab中的id
	domain_len : int #域名长度

class BatchInputFeature(NamedTuple):

	domain_names : List[str] #域名
	labels : Tensor  #是否为恶意地址
	char_ids : Tensor #每个字符在vocab中的id
	domain_lens : Tensor #每个域名的长度
