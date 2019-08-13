# PyTorch-自然语言处理-深度神经网络模型

Version 1.0 by KzXuan

**包含了PyTorch实现的CNN, RNN用于NLP领域的分类任务。**

全新设计的模块，优化大量代并降低使用复杂度，修复部分bug，使用mask作为序列长度标识。

即将包含：全新的序列标注支持，多GPU并行调参。

</br>

### 模型说明

* 环境：
  * Python >= 3.5
  * PyTorch >= 1.2.0

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


</br>

### pytorch模块说明

1. 网络层 **([layer.py](./dnnnlp/pytorch/layer.py))**

    * EmbeddingLayer(emb_matrix, emb_type='const')

      Embedding层，将词向量查询矩阵转化成torch内的可用变量，提供None/"const"/"variable"三种模式。

      **调用时传入原始输入即可，无需将输入转化成long类型。**

      ```python
      # 导入已有Embedding矩阵，训练中矩阵不可变
      emb_matrix = np.load("...")
      torch_emb_mat = layer.EmbeddingLayer(emb_matrix, 'const')
      # 查询下标获取完整inputs
      outputs = torch_emb_mat(inputs)
      ```

    * SoftmaxLayer(input_size, output_size)

      简单的Softmax层/全连接层，使用LogSoftmax作为激活函数。

      **调用时对tensor维度没有要求，调用后请使用NLLLoss计算损失。** 或可以使用Linear层和CrossEntropyLoss的组合计算损失。

      ```python
      # Softmax层进行二分类
      sl = layer.SoftmaxLayer(100, 2)
      # 调用
      prediction = sl(inputs)
      ```

    * CNNLayer(input_size, in_channels, out_channels, kernel_width, act_fun=nn.ReLU)

      封装的CNN层，支持最大池化和平均池化，支持自定义激活函数。

      **调用时需要传入一个四维的inputs来保证模型的正常运行，** 若传入的inputs为三维，会自动添加一个第二维，并在第二维上复制in_channels次。可选择输出模式"max"/"mean"/"all"来分别得到最大池化后的输出，平均池化后的输出或原始的全部输出。

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

    * RNNLayer(input_size, n_hidden, n_layer, drop_prob=0., bi_direction=True, mode="LSTM")

      封装的RNN层，支持tanh/LSTM/GRU，支持单/双向及多层堆叠。

      **调用时需要传入一个三维的inputs来保证模型的正常运行。** 可选择输出模式"all"/"last"来分别得到最后一层的全部隐层输出，或最后一层的最后一个时间步的输出。

      ```python
      # 创建堆叠式的两层GRU模型
      rnn_stack = nn.ModuleList()
      for _ in range(2):
          rnn_stack.append(
          layer.RNNLayer(input_size, n_hidden=50, n_layer=1, drop_prob=0.1, bi_direction=True, mode="GRU")
      )
      # 第一层GRU取全部输出
      outputs = inputs.reshape(-1, inputs.size(2), inputs.size(3))
      outputs = rnn_stack[0](outputs, seq_len_1, out_type='all')
      # 第二层GRU取最后一个时间步的输出
      outputs = outputs.reshape(inputs.size(0), inputs.size(1), -1)
      outputs = rnn_stack[1](outputs, seq_len_2, out_type='last')
      ```
2. 模型 **([model.py](./dnnnlp/pytorch/model.py))**

    * CNNModel(args, emb_matrix=None, kernel_widths=[2, 3, 4])

      常规CNN模型的封装，可以作为运行模块的输入模型，模型返回LogSoftmax后的预测概率。

      **支持多种卷积核宽度的同时设置，默认使用最大池化获得CNNLayer的输出，** 不支持层级结构。

      ```python
      # 模型初始化
      model = model.CNNModel(args, emb_matrix, [2, 3])
      # 调用时mask是可选参数
      pred = model(inputs, mask)
      ```

    * RNNModel(args, emb_matrix=None, n_hierarchy=1, n_layer=1, bi_direction=True, mode='LSTM')

      常规RNN模型的封装，可以作为运行模块的输入模型，模型返回LogSoftmax后的预测概率。

      **支持层次模型，默认在每一层次取最后一层的最后一个时间步的输出。**

      *Tip: 参数n_hierarchy用以控制模型的层次，每个层次会使得消除一个序列长度的维度，例如词-句子层次/句子-文档层次；参数n_layer用以控制每个层次内的RNN层数，每个RNN层将在pytorch内部直接叠加。*

      ```python
      # 模型初始化
      model = model.RNNModel(args, emb_matrix, n_hierarchy=2, mode='GRU')
      # 调用时mask是可选参数
      pred = model(inputs, mask)
      ```

3. 运行 **([exec.py](./dnnnlp/pytorch/exec.py))**

    * default_args()

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

    * exec(args)

      基类，提供基础函数，完成类内参数初始化。

    * classify(model, args, train_x, train_y, train_mask, test_x=None, test_y=None, test_mask=None, class_name=None, device_id=0)

      分类运行模块，提供完整的模型执行过程。

      * **需要传入一个有效的pytorch模型，该模型的返回值应是LogSoftmax后的预测概率。**

        *Tip: 若模型的返回值是Linear层的输出，可以修改实例化后的类内变量loss_function = nn.CrossEntropyLoss()。*

      * **对于训练数据，train_mask是必须值，若存在测试数据，test_mask也是必须值。**

        *Tip: 在NLP任务中，不存在mask的情况不多见，尽管dnnnlp提供的所有模型和层都支持无mask输入。若不存在mask，请构造全1的mask矩阵。*

      * **数据标签train_y和test_y使用标准形式（非one-hot形式）。**

      **调用时提供三种运行模式的接口：**

      (1) train_test()：训练-测试数据的调用函数

      (2) train_itself()：单一训练数据并使用本身进行测试的调用函数

      (3) cross_validation(fold=10)：k折交叉数据的调用函数

</br>

### utils模块说明

1. 评估 **([predict_eval.py](./dnnnlp/utils/predict_eval.py))**

    * prfacc1d(y_true, y_pred, one_hot=False, ndigits=4)

      **评估一维的预测概率。若标签和预测都为one-hot形式，则输入为二维。**

      返回一个包含所有评估指标的字典，各类别评估键值为'class0-p'/'class1-r'/'class2-f'等，宏平均的键值包含'macro-p'/'macro-r'/'macro-f'，微平均则使用'micro-X'，准确率的键值为'acc'。还包含各类别的统计数据，例如'correct'/'pred'/'real'。

    * prfacc2d(y_true, y_pred, mask=None, one_hot=False, ndigits=4)

      **评估二维的预测概率，通常指序列标注的预测概率，需要额外传入mask标记序列有效长度。若标签和预测都为one-hot形式，则输入为三维。**

      返回值同prfacc1d()。

2. 功能函数 **([easy_function.py](./dnnnlp/utils/easy_function.py))**

    * one_hot(arr, n_class=0)

      对numpy中的标准标签矩阵取one_hot形式。

      *Tip: 在pytorch中请使用torch.nn.functional.one_hot()函数。*

    * mold_fold(length, fold=10)

      按下标取余作为交叉验证的分块。

      返回一个列表，列表中的每个元素是一个三元组：(第几折，训练部分下标，测试部分下标)。

    * order_fold(length, fold=10)

      按序切分作为交叉验证的分块。

      返回值同mold_fold()。

    * len_to_mask(seq_len, max_seq_len)

      序列长度转mask矩阵，同时支持numpy和pytorch输入。

    * mask_to_len(mask)

      mask矩阵转序列长度，同时支持numpy和pytorch输入。

</br>

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

</br>

### 在GPU服务器上的使用

调试阶段，暂不支持，历史版本v0.12.3可用。
