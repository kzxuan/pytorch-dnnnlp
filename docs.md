# API

Version 1.0 by KzXuan

  * [Layer](#layer)
  * [Model](#model)
  * [Execution](#execution)
  * [Utility](#utility)

<br>

## Layer

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

> **SoftAttentionLayer(input_size)** <br>
>> forward(inputs, mask=None)

&nbsp;&nbsp;&nbsp;&nbsp;
简单的Attention层，为序列上的每一个位置生成权重并加和。

  * 调用时需要传入一个三位的inputs和二维的mask[可选]来保证模型的正常运行。

```python
# 注意力机制
sal = SoftAttentionLayer(input_size)
# 调用
outputs = sal(inputs)
```

<br>

> **CNNLayer(input_size, in_channels, out_channels, kernel_width, act_fun=nn.ReLU)** <br>
>> forward(inputs, mask=None, out_type='max')

&nbsp;&nbsp;&nbsp;&nbsp;
封装的CNN层，支持最大池化和平均池化，支持自定义激活函数。

  * **调用时需要传入一个四维的inputs和二维的mask[可选]来保证模型的正常运行。**
  * 若传入的inputs为三维，会自动添加一个第二维，并在第二维上复制in_channels次。
  * 可选择输出模式'max'/'mean'/'all'来分别得到最大池化后的输出，平均池化后的输出或原始的全部输出。

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

> **RNNLayer(input_size, n_hidden, n_layer=1, drop_prob=0., bi_direction=True, rnn_type='LSTM')** <br>
>> forward(inputs, mask=None, out_type='last')

&nbsp;&nbsp;&nbsp;&nbsp;
封装的RNN层，支持tanh/LSTM/GRU，支持单/双向及多层堆叠。

  * **调用时需要传入一个三维的inputs和二维的mask[可选]来保证模型的正常运行。**
  * 可选择输出模式'last'/'all'来分别得到最后一层的最后一个时间步的输出，或最后一层的全部隐层输出。

```python
# 创建堆叠式的两层GRU模型
rnn_stack = nn.ModuleList()
for _ in range(2):
    rnn_stack.append(
    layer.RNNLayer(input_size, n_hidden=50, n_layer=1, drop_prob=0.1, bi_direction=True, rnn_type='GRU')
)
# 第一层GRU取全部输出
outputs = inputs.reshape(-1, inputs.size(2), inputs.size(3))
outputs = rnn_stack[0](outputs, seq_len_1, out_type='all')
# 第二层GRU取最后一个时间步的输出
outputs = outputs.reshape(inputs.size(0), inputs.size(1), -1)
outputs = rnn_stack[1](outputs, seq_len_2, out_type='last')
```

<br>

> **MultiheadAttentionLayer(self, input_size, n_head=8, drop_prob=0.1)** <br>
>> forward(query, key=None, value=None, query_mask=None, key_mask=None)

&nbsp;&nbsp;&nbsp;&nbsp;
多头注意力层，封装pytorch官方提供的nn.MultiheadAttention方法。

  * 使用batch作为数据的第一维，与官方接口不同。
  * 分别为query和key提供mask选项。
  * **调用时需要传入一个三维的query和二维的query_mask[可选]来保证模型的正常运行。**
  * **单独提供query输入时，key和value复制query的值，key_mask复制query_mask的值。**
  * **提供query和key输入时，value复制key的值。**


```python
# 调用多头注意力机制
mal = layer.MultiheadAttentionLayer(input_size, n_head=8, drop_prob=0.1)
# 仅提供query和key以及对应的mask矩阵
outputs = mal(query, key, query_mask=query_mask, key_mask=key_mask)
```

<br>

> **TransformerLayer(input_size, n_head=8, feed_dim=None, drop_prob=0.1)** <br>
>> forward(query, key=None, value=None, query_mask=None, key_mask=None, out_type='first')

&nbsp;&nbsp;&nbsp;&nbsp;
封装的Transformer层，使用MultiheadAttentionLayer作为注意力机制层。

  * **调用遵循MultiheadAttentionLayer的规则。**
  * 可选择输出模式'first'/'all'来分别得到长度第一个位置的输出，或原始的全部输出。

```python
# 创建堆叠式的6层Transformer
trans = nn.ModuleList(
    [layer.TransformerLayer(args.emb_dim, n_head) for _  in range(6)]
)
# 前5层取全部输出
for li in range(5):
    outputs = trans[li](outputs, query_mask=mask, out_type='all')
# 最后一层取第一个位置的输出
outputs = trans[5](outputs, query_mask=mask, out_type='first')
```

<br>

## Model

> **CNNModel(args, emb_matrix=None, kernel_widths=[2, 3, 4])** <br>
>> forward(inputs, mask=None)

&nbsp;&nbsp;&nbsp;&nbsp;
常规CNN模型的封装，模型返回LogSoftmax后的预测概率。

  * **支持多种卷积核宽度的同时设置。**
  * 默认使用最大池化获得CNNLayer的输出。
  * 不支持层次结构。

```python
# 模型初始化
model = model.CNNModel(args, emb_matrix, [2, 3])
# 调用时mask是可选参数
pred = model(inputs, mask)
```

<br>

> **RNNModel(args, emb_matrix=None, n_hierarchy=1, n_layer=1, bi_direction=True, rnn_type='LSTM', use_attention=False)** <br>
>> forward(inputs, mask=None)

&nbsp;&nbsp;&nbsp;&nbsp;
常规RNN模型的封装，模型返回LogSoftmax后的预测概率。

  * **支持层次结构。**
  * 默认使用SoftAttentionLayer作为注意力层。
  * 若不使用注意力机制，默认在每一层次取最后一层的最后一个时间步的输出。
  * 若使用注意力机制，默认在每一层次取最后一层的全部输出，再通过注意力层。

```python
# 模型初始化
model = model.RNNModel(args, emb_matrix, n_hierarchy=2, rnn_type='GRU')
# 调用时mask是可选参数
pred = model(inputs, mask)
```

*Tip: 参数n_hierarchy用以控制模型的层次，每个层次会使得消除一个序列长度的维度，例如词-句子层次/句子-文档层次；参数n_layer用以控制每个层次内的RNN层数，每个RNN层将在pytorch内部直接叠加。*

<br>

> **TransformerModel(args, emb_matrix=None, n_layer=6, n_head=8)** <br>
>> forward(inputs, mask=None)

&nbsp;&nbsp;&nbsp;&nbsp;
常规Transformer模型的封装，模型返回LogSoftmax后的预测概率。

  * **使用堆叠式的多层TransformerLayer。**
  * 仅提供一个inputs输入，即TransformerLayer中的query/key/value均会被初始化为inputs。
  * 不支持层次结构。

```python
# 模型初始化
model = model.TransformerModel(args, emb_matrix, n_layer=12, n_head=8)
# 调用时mask是可选参数
pred = model(inputs, mask)
```

<br>

## Execution

> **set_seed(seed=100)**

&nbsp;&nbsp;&nbsp;&nbsp;
设置pytorch的参数初始化随即种子。

<br>

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

    (1) train_test(device_id=0)：训练-测试数据的调用函数

    (2) train_itself(device_id=0)：单一训练数据并使用本身进行测试的调用函数

    (3) cross_validation(fold=10, device_id=0)：k折交叉数据的调用函数

    *Tip: device_id可以用来选择模型的运行硬件，-1表示使用CPU运行，0/1/...表示使用第几块GPU运行。*

<br>

> **average_several_run(run_func, args, n_times=4, n_paral=2, \*\*run_params)**

&nbsp;&nbsp;&nbsp;&nbsp;
多GPU并行同一模型，并获取多次运行后的均值。

  * run_func接受某一运行模块的运行函数。
  * **使用n_paral设置并行的GPU数（或CPU内的进程数），是否使用GPU以及单次运行模型所需要的GPU数仍然由args.n_gpu设定。**
  * 模型运行次数n_times需为n_paral的整数倍。

```python
# 初始化一个分类运行模块
nn = Classify(model, args, train_x, train_y, train_mask)
# 4GPU并行共运行模型8次
avg_socre = average_several_run(nn.cross_validation, args, n_times=8, n_paral=4, fold=5)
```

<br>

> **grid_search(exec_class, run_func, args, params_search, n_paral=2, \*\*run_params)**

&nbsp;&nbsp;&nbsp;&nbsp;
多GPU并行同一模型，并获取参数网格搜索后的结果。<br>
&nbsp;&nbsp;&nbsp;&nbsp;
**WARNING** 存在多进程导致GPU显存溢出的可能性，尚不清楚原因，重新运行或许可以解决。

  * exec_class接受实例化后的某一运行模块，run_func接受该模块的运行函数。
  * **使用n_paral设置并行的GPU数（或CPU内的进程数）。**

```python
# 初始化一个分类运行模块
nn = Classify(model, args, train_x, train_y, train_mask)
# 需要搜索的参数
params_search = {'learning_rate': [0.1, 0.01], 'n_hidden': [50, 100]}
# 2GPU并行网格搜索
max_score = grid_search(nn, nn.train_test, args, params_search)
```

<br>

## Utility

> **prfacc(y_true, y_pred, mask=None, one_hot=False, ndigits=4, tabular=False)**

&nbsp;&nbsp;&nbsp;&nbsp;
评估预测概率与真实标签，获得所有评估指标。

  * 封装sklearn中的classification_report方法。（对sklearn提供的macro-f1计算方法尚存争议）
  * **标签和预测可以是任意维度，但必须维度相等。**
  * **对于序列标注等任务，可以传入mask矩阵标记序列有效长度。**
  * **支持输入格式为list/np.ndarray/torch.tensor。**
  * 标签和预测可以同时为one-hot形式。
  * 若tabular为True，返回一个字符串形式的评估表格。
  * 若tabular为False，返回一个包含所有评估指标的字典。

```python
# 评估预测标签
y_true = [1, 1, 1, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 1, 1, 0, 0, 0]
evals = prfacc(y_true, y_pred, tabular=False)
# evals
{
    'accuracy': 0.625,
    'class0': {'precision': 0.5, 'recall': 0.6667, 'f1-score': 0.5714, 'support': 3},
    'class1': {'precision': 0.75, 'recall': 0.6, 'f1-score': 0.6667, 'support': 5},
    'macro': {'precision': 0.625, 'recall': 0.6333, 'f1-score': 0.619, 'support': 8},
    'weighted': {'precision': 0.6562, 'recall': 0.625, 'f1-score': 0.631, 'support': 8}
}
```

*Tip: 可以使用pandas.DataFrame(evals, dtype=str).transpose()将字典重新制表。*

<br>

> **average_prfacc(\*evals, ndigits=4)**

&nbsp;&nbsp;&nbsp;&nbsp;
为多个评估结果取均值。

<br>

> **maximum_prfacc(\*evals, eval_matric='accuracy')**

&nbsp;&nbsp;&nbsp;&nbsp;
筛选多个评估结果中，指定评估指标最大的一个。

  * 对于指定评估指标为'macro'/'class0'等情况，取该指标下'f1-score'最大的评估结果。

<br>

> **display_prfacc(\*eval_metrics, sep='|', verbose=2)** <br>
>> row(evals)

&nbsp;&nbsp;&nbsp;&nbsp;
在模型运行的每个iteration，以表格形式输出评估结果变化。

  * 当前迭代iter、损失loss和准确率accuracy为必然输出。
  * 可以添加其它需要输出的指定评估指标，例如'macro'/'class1'。
  * 利用verbose等级控制表格是否输出。

<br>

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

[Top](#API)
