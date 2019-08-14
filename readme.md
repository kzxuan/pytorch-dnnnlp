## PyTorch-自然语言处理-深度神经网络模型

Version 1.0 by KzXuan

**包含了PyTorch实现的CNN, RNN用于NLP领域的分类任务。**

* 全新设计的两大模块
* 优化大量代码逻辑并降低使用复杂度
* 更少的内存占用量
* 使用mask作为序列长度标识
* 部分bug修复

即将包含：新的序列标注支持，多GPU并行调参。

<br>

## 说明

* 环境：

  python >= 3.5 & pytorch >= 1.2.0

* 超参数说明：

  | 参数名        | 类型  | 默认值 | 说明                                                 |
  | ------------- | ----- | ------ | ---------------------------------------------------- |
  | n_gpu         | int   | 1      | 使用GPU的数量（0表示不使用GPU加速）                  |
  | space_turbo   | bool  | True   | 利用更多的GPU显存进行加速                            |
  | data_shuffle  | bool  | Ture   | 是否打乱数据进行训练或测试                           |
  | emb_type      | str   | None   | 使用None/'const'/'variable'表示Embedding模式         |
  | emb_dim       | int   | 300    | Embedding维度（输入为特征时表示特征的维度）          |
  | n_class       | int   | 2      | 分类的目标类数                                       |
  | n_hidden      | int   | 50     | 隐层节点数，或CNN的输出通道数                        |
  | learning_rate | float | 0.01   | 学习率                                               |
  | l2_reg        | float | 1e-6   | L2正则                                               |
  | batch_size    | int   | 128    | 批量大小                                             |
  | iter_times    | int   | 30     | 迭代次数                                             |
  | display_step  | int   | 2      | 迭代过程中显示输出的间隔迭代次数                     |
  | drop_prob     | float | 0.1    | Dropout比例                                          |
  | eval_metric   | str   | 'acc'  | 使用'acc'/'macro'/'micro'/'class1'等设定模型评判标准 |

<br>

## pytorch模块说明

### 网络层 ([layer.py](./dnnnlp/pytorch/layer.py))

> **EmbeddingLayer(emb_matrix, emb_type='const')** <br>
>> forward(inputs)

&nbsp;&nbsp;&nbsp;&nbsp;
Embedding层，将词向量查询矩阵转化成torch内的可用变量。

  * 提供None/"const"/"variable"三种模式，对应无查询矩阵/不可调/可调模式。
  * **调用时传入原始输入即可，无需将输入转化成long类型。**

```python
# 导入已有Embedding矩阵，训练中矩阵不可变
emb_matrix = np.load("...")
torch_emb_mat = layer.EmbeddingLayer(emb_matrix, 'const')
# 查询下标获取完整inputs
outputs = torch_emb_mat(inputs)
```

<br>

> **SoftmaxLayer(input_size, output_size)** <br>
>> forward(inputs)

&nbsp;&nbsp;&nbsp;&nbsp;
简单的Softmax层/全连接层，使用LogSoftmax作为激活函数。

  * 调用时对tensor维度没有要求，调用后请使用nn.NLLLoss计算损失。

```python
# Softmax层进行二分类
sl = layer.SoftmaxLayer(100, 2)
# 调用
prediction = sl(inputs)
```

*Tip: 也可以在模型预测层选择nn.Linear层，并配合nn.CrossEntropyLoss来计算损失。*

<br>

> **CNNLayer(input_size, in_channels, out_channels, kernel_width, act_fun=nn.ReLU)**

&nbsp;&nbsp;&nbsp;&nbsp;
封装的CNN层，支持最大池化和平均池化，支持自定义激活函数。

  * **调用时需要传入一个四维的inputs来保证模型的正常运行。** 若传入的inputs为三维，会自动添加一个第二维，并在第二维上复制in_channels次。
  * 可选择输出模式"max"/"mean"/"all"来分别得到最大池化后的输出，平均池化后的输出或原始的全部输出。

```python
# 创建卷积核宽度分别为2、3、4的且通道数为50的CNN集合
cnn_set = nn.ModuleList()
for kw in range(2, 5):
    cnn_set.append(
        layer.CNNLayer(emb_dim, in_channels=1, out_channels=50, kernel_width=kw)
)
# 将调用后的结果进行拼接
outputs = torch.cat([c(inputs, seq_len, out_type='max') for c in cnn_set], -1)
```

<br>

> **RNNLayer(input_size, n_hidden, n_layer, drop_prob=0., bi_direction=True, rnn_type="LSTM")**

>> forward()

&nbsp;&nbsp;&nbsp;&nbsp;
封装的RNN层，支持tanh/LSTM/GRU，支持单/双向及多层堆叠。

  * **调用时需要传入一个三维的inputs来保证模型的正常运行。**
  * 可选择输出模式"all"/"last"来分别得到最后一层的全部隐层输出，或最后一层的最后一个时间步的输出。

```python
# 创建堆叠式的两层GRU模型
rnn_stack = nn.ModuleList()
for _ in range(2):
    rnn_stack.append(
    layer.RNNLayer(input_size, n_hidden=50, n_layer=1, drop_prob=0.1, bi_direction=True, rnn_type="GRU")
)
# 第一层GRU取全部输出
outputs = inputs.reshape(-1, inputs.size(2), inputs.size(3))
outputs = rnn_stack[0](outputs, seq_len_1, out_type='all')
# 第二层GRU取最后一个时间步的输出
outputs = outputs.reshape(inputs.size(0), inputs.size(1), -1)
outputs = rnn_stack[1](outputs, seq_len_2, out_type='last')
```

<br>

> **MultiheadAttentionLayer(self, input_size, n_head=8, drop_prob=0.1)**

&nbsp;&nbsp;&nbsp;&nbsp;
多头注意力层，封装pytorch官方提供的nn.MultiheadAttention方法。

  * 使用batch作为数据的第一维。
  * 分别为query和key提供mask选项。
  * **单独提供query输入时，key和value复制query的值，key_mask复制query_mask的值。**
  * **提供query和key输入时，value复制key的值。**


```python
# 利用多头注意力机制
mal = layer.MultiheadAttentionLayer(input_size, n_head=8, drop_prob=0.1)
# 仅提供query和key以及对应的mask矩阵
outputs = mal(query, key, query_mask=query_mask, key_mask=key_mask)
```

<br>

> **TransformerLayer(input_size, n_head=8, feed_dim=None, drop_prob=0.1)**

&nbsp;&nbsp;&nbsp;&nbsp;
封装的Transformer层，使用MultiheadAttentionLayer作为注意力机制层。

  *

<br>

### 模型 ([model.py](./dnnnlp/pytorch/model.py))

> **CNNModel(args, emb_matrix=None, kernel_widths=[2, 3, 4])**

&nbsp;&nbsp;&nbsp;&nbsp;
常规CNN模型的封装，模型返回LogSoftmax后的预测概率。

  * **支持多种卷积核宽度的同时设置。**
  * 默认使用最大池化获得CNNLayer的输出。
  * 不支持层级结构。

```python
# 模型初始化
model = model.CNNModel(args, emb_matrix, [2, 3])
# 调用时mask是可选参数
pred = model(inputs, mask)
```

<br>

> **RNNModel(args, emb_matrix=None, n_hierarchy=1, n_layer=1, bi_direction=True, mode='LSTM')**

&nbsp;&nbsp;&nbsp;&nbsp;
常规RNN模型的封装，模型返回LogSoftmax后的预测概率。

  * **支持层次模型。**
  * 默认在每一层次取最后一层的最后一个时间步的输出。

```python
# 模型初始化
model = model.RNNModel(args, emb_matrix, n_hierarchy=2, mode='GRU')
# 调用时mask是可选参数
pred = model(inputs, mask)
```

*Tip: 参数n_hierarchy用以控制模型的层次，每个层次会使得消除一个序列长度的维度，例如词-句子层次/句子-文档层次；参数n_layer用以控制每个层次内的RNN层数，每个RNN层将在pytorch内部直接叠加。*

<br>

### 运行 ([exec.py](./dnnnlp/pytorch/exec.py))

> **default_args()**

&nbsp;&nbsp;&nbsp;&nbsp;
初始化所有超参数，并返回参数集。

```python
def default_args():
    # ...
    return args

args = default_args()

# 程序内修改参数
args.n_hidden = 100
args.batch_size = 32
```

```bash
# 在命令行中传递参数，与程序内修改参数互斥
> python3 demo.py --n_hidden 100 --batch_size 32
```

<br>

> **Classify(model, args, train_x, train_y, train_mask, test_x=None, test_y=None, test_mask=None, class_name=None, device_id=0)**

&nbsp;&nbsp;&nbsp;&nbsp;
分类运行模块，提供完整的模型执行过程。

  * **需要传入一个有效的pytorch模型，该模型的返回值应是LogSoftmax后的预测概率。**

    *Tip: 若模型的返回值是Linear层的输出，可以修改实例化后的类内变量loss_function = nn.CrossEntropyLoss()。*

  * **对于训练数据，train_mask是必须值，若存在测试数据，test_mask也是必须值。**

    *Tip: 在NLP任务中，不存在mask的情况不多见，尽管dnnnlp提供的所有模型和层都支持无mask输入。若不存在mask，请构造全1的mask矩阵。*

  * **数据标签train_y和test_y使用标准形式（非one-hot形式）。**

  * **调用时提供三种运行模式的接口：**

    (1) train_test()：训练-测试数据的调用函数

    (2) train_itself()：单一训练数据并使用本身进行测试的调用函数

    (3) cross_validation(fold=10)：k折交叉数据的调用函数

<br>

## utils模块说明

### 评估 ([predict_eval.py](./dnnnlp/utils/predict_eval.py))

> **prfacc1d(y_true, y_pred, one_hot=False, ndigits=4)**

&nbsp;&nbsp;&nbsp;&nbsp;
评估一维标签的预测概率。

  * **若标签和预测都为one-hot形式，则输入为二维。**
  * 返回一个包含所有评估指标的字典，各类别评估键值为'class0-p'/'class1-r'/'class2-f'等，宏平均的键值包含'macro-p'/'macro-r'/'macro-f'，微平均则使用'micro-X'，准确率的键值为'acc'。还包含各类别的统计数据，例如'correct'/'pred'/'real'。

<br>

> **prfacc2d(y_true, y_pred, mask=None, one_hot=False, ndigits=4)**

&nbsp;&nbsp;&nbsp;&nbsp;
评估二维的预测概率，通常指序列标注的预测概率。

  * 需要额外传入mask标记序列有效长度。
  * 若标签和预测都为one-hot形式，则输入为三维。
  * 返回值同prfacc1d()。

<br>

### 功能函数 ([easy_function.py](./dnnnlp/utils/easy_function.py))

> **one_hot(arr, n_class=0)**

&nbsp;&nbsp;&nbsp;&nbsp;
对numpy中的标准标签矩阵取one_hot形式。

*Tip: 在pytorch中请使用torch.nn.functional.one_hot()函数。*

<br>

> **mold_fold(length, fold=10)**

&nbsp;&nbsp;&nbsp;&nbsp;
按下标取余作为交叉验证的分块。

  * 返回一个列表，列表中的每个元素是一个三元组：(第几折，训练部分下标，测试部分下标)。

<br>

> **order_fold(length, fold=10)**

&nbsp;&nbsp;&nbsp;&nbsp;
按序切分作为交叉验证的分块。

  * 返回值同mold_fold()。

<br>

> **len_to_mask(seq_len, max_seq_len)**

&nbsp;&nbsp;&nbsp;&nbsp;
序列长度转mask矩阵，同时支持numpy和pytorch输入。

<br>

> **mask_to_len(mask)**

&nbsp;&nbsp;&nbsp;&nbsp;
mask矩阵转序列长度，同时支持numpy和pytorch输入。

<br>

### 模型功能及使用

* 分类

  ````python
  from dnnnlp.pytorch.model import RNNModel
  from dnnnlp.pytorch.exec import default_args, Classify

  emb_mat = np.array([...])
  args = default_args()

  model = RNNModel(args)

  class_name = ['support', 'deny', 'query', 'comment']
  nn = Classify(model, args, train_x, train_y, train_mask, test_x, test_y, test_mask, emb_matrix, class_name)
  nn.cross_validation(fold=10)
  ````

<br>

### 在GPU服务器上的使用

调试阶段，暂不支持，历史版本v0.12.3可用。
