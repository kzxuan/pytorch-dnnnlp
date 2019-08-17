# PyTorch-深度神经网络模型-自然语言处理

Version 1.0 by KzXuan

**包含了PyTorch实现的CNN & RNN & Transformer用于NLP领域的分类任务。**

* 全新的模块设计
* 优化大量代码逻辑并降低使用复杂度
* 使用mask作为序列长度标识
* 多GPU并行的方式调参
* 部分bug修复

即将包含：新的序列标注支持

<br>

## 说明

* 环境：

  python >= 3.5 & pytorch >= 1.2.0

* API文档

  * [层](./docs.md#层) - [layer.py](./dnnnlp/layer.py)
  * [模型](./docs.md#模型) -  [model.py](./dnnnlp/model.py)
  * [运行模块](./docs.md#运行模块) - [exec.py](./dnnnlp/exec.py)
  * [工具](./docs.md#工具) - [utils.py](./dnnnlp/utils.py)

* 超参数说明：

  | 参数名         | 类型   | 默认值       | 说明                                                 |
  | ------------- | ----- | ----------- | ---------------------------------------------------- |
  | n_gpu         | int   | 1           | 使用GPU的数量（0表示不使用GPU加速）                    |
  | space_turbo   | bool  | True        | 利用更多的GPU显存进行加速                            |
  | data_shuffle  | bool  | Ture        | 是否打乱数据进行训练或测试                           |
  | emb_type      | str   | None        | 使用None/'const'/'variable'表示Embedding模式         |
  | emb_dim       | int   | 300         | Embedding维度（输入为特征时表示特征的维度）          |
  | n_class       | int   | 2           | 分类的目标类数                                       |
  | n_hidden      | int   | 50          | 隐层节点数，或CNN的输出通道数                        |
  | learning_rate | float | 0.01        | 学习率                                               |
  | l2_reg        | float | 1e-6        | L2正则                                               |
  | batch_size    | int   | 128         | 批量大小                                             |
  | iter_times    | int   | 30          | 迭代次数                                             |
  | display_step  | int   | 2           | 迭代过程中显示输出的间隔迭代次数                         |
  | drop_prob     | float | 0.1         | Dropout比例                                          |
  | eval_metric   | str   | 'accuracy'  | 使用'accuracy'/'macro'/'class1'等设定模型评判标准       |

<br>

## 使用示例

* 分类

  ````python
  from dnnnlp.model import RNNModel
  from dnnnlp.exec import default_args, Classify

  emb_mat = np.array([...])
  args = default_args()

  model = RNNModel(args, emb_mat, bi_direction=False, rnn_type='GRU', use_attention=True)

  nn = Classify(model, args, train_x, train_y, train_mask, test_x, test_y, test_mask)
  nn.train_test()
  ````

* 并行

  ````python
  from dnnnlp.model import CNNModel
  from dnnnlp.exec import default_args, Classify, average_several_run

  emb_mat = np.array([...])
  args = default_args()

  model = CNNModel(args, emb_mat, kernel_widths=[2, 3, 4])

  nn = Classify(model, args, train_x, train_y, train_mask)
  avg_scores = average_several_run(nn.cross_validation, args, n_times=8, n_paral=4, fold=5)
  ````

* 网格搜索

  ````python
  from dnnnlp.model import TransformerModel
  from dnnnlp.exec import default_args, Classify, grid_search

  emb_mat = np.array([...])
  args = default_args()

  model = TransformerModel(args, n_layer=12, n_head=8)

  nn = Classify(model, args, train_x, train_y, train_mask)
  params_search = {'learning_rate': [0.1, 0.01], 'n_hidden': [50, 100]}
  max_scores = grid_search(nn, nn.train_test, args, params_search)
  ````

<br>

## 在GPU服务器上的使用

```python
from dnnnlp.exec import Classify
from dnnnlp.model import RNNModel
from dnnnlp.layer import TransformerLayer
from dnnnlp import utils, layer, model, exec
```

<br>

[返回顶部](#PyTorch-深度神经网络模型-自然语言处理)
