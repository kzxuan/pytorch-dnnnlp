## PyTorch-自然语言处理-深度神经网络模型

Version 1.0 by KzXuan

**包含了PyTorch实现的CNN, RNN, Transformer用于NLP领域的分类任务（序列标注任务将在后期重新支持）。**



</br>

#### 模型说明

* 环境：
  * Python >= 3.5
  * PyTorch >= 0.4.1

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

#### pytorch模块说明

1. 封装网络层 **([layer.py](./dnnnlp/pytorch/layer.py))**

   * EmbeddingLayer(emb_matrix, emb_type='const')

     Embedding层，将词向量查询矩阵转化成torch内的可用变量，提供"const"/"variable"两种模式。

     ```python
     # 导入已有Embedding矩阵，训练中矩阵不可变
     emb_matrix = np.load("...")
     torch_emb_mat = layer.EmbeddingLayer(emb_matrix, 'const')
     # 查询下标获取完整inputs
     outputs = torch_emb_mat(inputs)
     ```

   * SoftmaxLayer(input_size, output_size)

     简单的Softmax层/全连接层，使用LogSoftmax作为激活函数，期望使用NLLLoss计算损失。

     ```python
   # Softmax层进行二分类
     sl = layer.SoftmaxLayer(100, 2)
     
     prediction = sl(inputs)
     ```
  
   * CNNLayer(input_size, in_channels, out_channels, kernel_width, act_fun=nn.ReLU)

     封装的CNN层，支持最大池化和平均池化，支持自定义激活函数。

     **调用时需要传入一个四维的inputs来保证模型的正常运行**，若传入的inputs为三维，会自动添加一个第二维，并在第二维上复制in_channels次。可选择输出模式"max"/"mean"/"all"来分别得到最大池化后的输出，平均池化后的输出或原始的全部输出。
   
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
   
     **调用时需要传入一个三维的inputs来保证模型的正常运行。**可选择输出模式"all"/"last"来分别得到最后一层的全部隐层输出，或最后一层的最后一个时间步的输出。
   
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
   
3. 封装模型 **([model.py](./dnnnlp/pytorch/model.py))**

   * CNNModel(args, emb_matrix=None, kernel_widths=[2, 3, 4])

     常规CNN模型的封装，支持多种卷积核宽度的同时输入，不支持层级结构，模型返回预测概率。

   * RNNModel(args, emb_matrix=None, n_hierarchy=1, n_layer=1, bi_direction=True, mode='LSTM')

     常规RNN层次模型的封装，支持多层次的分类，模型返回预测概率。

4. 运行模块 **([exec.py](./pytorch/exec.py))**

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
   
  基础运行模块，提供基础函数，完成类内参数初始化。
   
   * classify(args, train_x, train_y, train_mask, test_x=None, test_y=None, test_mask=None, class_name=None, device_id=0)
   
     分类模块基类，提供完整的程序执行过程，提供三种运行模式：
   
     (1) train_test()：训练-测试数据的调用函数
   
     (2) train_itself()：单一训练数据并使用本身进行测试的调用函数
   
     (3) cross_validation(fold=10)：k折交叉数据的调用函数
   
   * CNNClassify(args, train_x, train_y, train_mask, test_x=None, test_y=None, test_mask=None, emb_matrix=None, kernel_widths=[2, 3, 4], class_name=None, device_id=0)
   
     **使用CNN分类的执行模块。**
   
   * RNNClassify(args, train_x, train_y, train_mask, test_x=None, test_y=None, test_mask=None, emb_matrix=None, n_hierarchy=1, n_layer=1, bi_direction=True, mode='LSTM', class_name=None, device_id=0)
   

</br>

#### 模型功能及使用

* 分类

  ```python
  from dnnnlp.pytorch.exec import default_args, RNNClassify
  
  emb_mat = np.array([...])
  args = default_args(data_dict)
  class_name = ['support', 'deny', 'query', 'comment']
  nn = RNNClassify(args, train_x, train_y, train_mask, test_x, test_y, test_mask, emb_matrix, mode='GRU', class_name=class_name)
  nn.cross_validation(fold=10)
  ```
  

</br>

#### 在GPU服务器上的使用

调试阶段，暂不支持，v0.12可用。

