# DGA-Domain-Detection
基于pytorch LSTM的恶意域名检测实例 

### 1.下载数据

链接：https://pan.baidu.com/s/1DQffQxJxycgL3H9WXwJiZA 
提取码：gqpa

放入data/中.

w.csv : 白名单数据 共100万

b.csv : 黑名单数据 共1.5万

### 2.训练模型

运行Train.py

在Config.py中调参

### 3.预测

运行predict.py

### 4.效果

目前未精调参数。在model selection中比较了两组参数的效果

目前F1值最好在0.849