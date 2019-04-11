## PyTorch简单深度神经网络模型

Version 0.12 by KzXuan

**包含了PyTorch实现的CNN, RNN, Transformer用于NLP领域的分类及序列标注任务。**

相比TensorFlow的静态图模型，PyTorch拥有更为原生的Python语言写法，默认支持动态建图。

在同数据集同模型同参数的设置下，一份较大的数据进行十折交叉用时PyTorch 4m40s VS TensorFlow 20m06s。

</br>

#### 模型说明

* 环境：
  * Python >= 3.5
  * PyTorch >= 0.4.1

* 超参数说明：

  | 参数名         | 类型  | 默认值 | 说明                                                  | 模型  |
  | -------------- | ----- | ------ | ----------------------------------------------------- | ----- |
  | n_gpu          | int   | 1      | 使用GPU的数量（0表示不使用GPU加速）                   | C R T |
  | space_turbo    | bool  | True   | 利用更多的GPU显存进行加速                             | C R T |
  | data_shuffle   | bool  | Ture   | 是否打乱数据进行训练或测试                            | C R T |
  | emb_type       | str   | None   | 使用None/'const'/'variable'/'random'表示Embedding模式 | C R T |
  | emb_dim        | int   | 300    | Embedding维度（输入为特征时表示特征的维度）           | C R T |
  | n_class        | int   | 2      | 分类的目标类数                                        | C R T |
  | n_hidden       | int   | 50     | 隐层节点数，或CNN的输出通道数                         | C R T |
  | learning_rate  | float | 0.01   | 学习率                                                | C R T |
  | l2_reg         | float | 1e-6   | L2正则                                                | C R T |
  | batch_size     | int   | 128    | 批量大小                                              | C R T |
  | iter_times     | int   | 30     | 迭代次数                                              | C R T |
  | display_step   | int   | 2      | 迭代过程中显示输出的间隔迭代次数                      | C R T |
  | drop_prob      | float | 0.1    | Dropout比例                                           | C R T |
  | score_standard | str   | 'Acc'  | 使用'Ma-F'/…/'C1-R'/'C1-F'/'Acc'等设定模型评判标准    | C R T |
  | rnn_type       | str   | 'LSTM' | 使用'tanh'/'LSTM'/'GRU'设定RNN的核心模型              | R     |
  | bi_direction   | bool  | True   | 双向/单向RNN                                          | R     |
  | use_attention  | bool  | False  | 是否使用注意力机制（默认在每一层次的RNN上添加）       | R     |
  | n_layer        | int   | 1      | 每个层次的RNN层数或Transformer的层数                  | R T   |
  | n_head         | Int   | 8      | Transformer模型的注意力头数                           | T     |

* 数据要求：

  **构建data_dict并送入模型，data_dict为数据字典**，其中包含：

  | 键值 | 类型                 | 说明                                                         | 必须值 |
  | ---- | -------------------- | ------------------------------------------------------------ | ------ |
  | x    | np.array             | 训练集输入数据                                               | 是     |
  | y    | np.array             | 训练集标签                                                   | 是     |
  | len  | list [np.array, ...] | 训练集序列长度，从前往后表示模型从下到上每一个层级的序列长度 | 是     |
  | tx   | np.array             | 测试集输入数据                                               | 否     |
  | ty   | np.array             | 测试集标签                                                   | 否     |
  | tlen | list [np.array, ...] | 测试集序列长度，内容同训练集                                 | 否     |

</br>

#### 代码说明

1. 参数和基类 **([base.py](./pytorch/base.py))**

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

   * base(args)

     基类，接受args参数集，初始化模型参数，并包含部分基本函数，集成多次运行取平均、参数网格搜索等功能（详见<模型功能与使用>）。

     ```python
     # 创建训练数据集
     train_loader = base.create_data_loader(
         torch.tensor(data_dict['x'], dtype=torch.float, device=torch.device("cuda:0")),
         torch.tensor(data_dict['y'], dtype=torch.int, device=torch.device("cuda:0")),
         *[torch.tensor(e, dtype=torch.int, device=torch.device("cuda:0")) for e in data_dict['len']]
     )
     
     # 使用下标交叉验证
     for nf, train, test in base.mod_fold(data_dict['x'].shape[0], fold=10):
         pass
     
     # 使用按序交叉验证
     for nf, train, test in base.order_fold(data_dict['x'].shape[0], fold=10):
         pass
     ```

2. 封装网络层 **([layer.py](./pytorch/layer.py))**

   封装的类都可以脱离项目提供的构造环境来单独运行，且均提供参数初始化函数，需要在类实例化后调用。

   * embedding_layer(emb_matrix, emb_type='const')

     Embedding层，将词向量查询矩阵转化成torch内的可用变量，提供"const"/"variable"/"random"三种模式。

     ```python
     # 导入已有Embedding矩阵，训练中矩阵不可变
     emb_matrix = np.load("...")
     torch_emb_mat = layer.embedding_layer(emb_matrix, 'const')
     # 查询下标获取完整inputs
     outputs = torch_emb_mat(inputs)
     ```

   * positional_embedding_layer(emb_dim, max_len=512)

     位置Embedding层，为序列创建位置信息。

     ```python
     # 为最大长度为140的序列创建50维位置Embedding
     pel = layer.positional_embedding_layer(50, max_len=140)
     # 获取当前输入的位置Embedding
     pos_emb = per(inputs)
     ```

   * self_attention_layer(n_hidden)

     自注意力机制层，接受隐层节点数n_hidden参数。

     ```python
     # 创建自注意力层
     sal = layer.self_attention_layer(n_hidden=50)
     # 参数自定义初始化（在有需求的情况下调用）
     sal.init_weights()
     # 调用时若需要使用长度信息，seq_len必须是一维向量，下同
     outputs = sal(inputs, seq_len)
     ```

   * CNN_layer(input_size, in_channels, out_channels, kernel_width, stride=1)

     封装的CNN层，支持最大池化和平均池化。

     **调用时需要传入一个四维的inputs来保证模型的正常运行**，若传入的inputs为三维，会自动添加一个第二维，并在第二维上复制in_channels次。可选择输出模式"max"/"mean"/"all"来分别得到最大池化后的输出，平均池化后的输出或原始的全部输出。

     ```python
     # 创建卷积核宽度分别为2、3、4的且通道数为50的CNN集合
     cnn_set = nn.ModuleList()
     for kw in range(2, 5):
         cnn_set.append(
             layer.CNN_layer(emb_dim, in_channels=1, out_channels=50, kernel_width=kw, stride=1)
         )
     # 将调用后的结果进行拼接
     outputs = torch.cat([c(inputs, seq_len, out_type='max') for c in cnn_set], -1)
     ```

   * RNN_layer(input_size, n_hidden, n_layer, drop_prob, bi_direction, mode="LSTM")

     封装的RNN层，支持tanh/LSTM/GRU，支持单/双向及多层堆叠。

     **调用时需要传入一个三维的inputs来保证模型的正常运行。**可选择输出模式"all"/"last"来分别得到最后一层的全部隐层输出，或最后一层的最后一个时间步的输出。

     ```python
     # 创建堆叠式的两层GRU模型
     rnn_stack = nn.ModuleList()
     for _ in range(2):
         rnn_stack.append(
             layer.RNN_layer(input_size, n_hidden=50, n_layer=1, drop_prob=0.1, bi_direction=True, mode="GRU")
         )
     # 第一层GRU取全部输出
     outputs = torch.reshape(inputs, [-1, inputs.size(2), inputs.size(3)])
     outputs = rnn_stack[0](outputs, seq_len_1, out_type='all')
     # 第二层GRU取最后一个时间步的输出
     outputs = torch.reshape(outputs, [inputs.size(0), inputs.size(1), -1])
     outputs = rnn_stack[1](outputs, seq_len_2, out_type='last')
     ```

   * softmax_layer(n_in, n_out)

     简单的Softmax层/全连接层。

     ```python
     # Softmax层进行二分类
     sl = layer.softmax_layer(100, 2)
     sl.init_weights()
     
     prediction = sl(inputs)
     ```

   * multi_head_attention_layer(query_size, key_size=None, value_size=None, n_hidden=None, n_head=8)

     多头注意力层，Transformer内的注意力操作。若key或value的初始维度和初始值未给出，将自动使用query作为key或value。

     ```python
     # 单个输入query使用多头注意力
     mhal = layer.multi_head_attention_layer(query_size, n_hidden=50, n_head=16)
     
     outputs = mhal(query, seq_len=seq_len)
     ```

   * transformer_layer(input_size, n_hidden, n_head, drop_prob=0.1)

     封装的Transformer层，支持多头注意力、任意位提取输出。

     调用时直接取inputs同时作为query/key/value，可以选择提取序列输出的某一个位置作为整个transformer层的输出。

     ```python
     # 创建堆叠式的三层Transformer模型
     trans_stack = nn.ModuleList()
     for _ in range(3):
         trans_stack.append(layer.transformer_layer(emb_dim, n_hidden, n_head, drop_prob))
     # 前两层取全部位置的输出
     for i in range(2):
         outputs = trans_stack[i](outputs, seq_len, get_index=None)
     # 第三层取第一个位置的输出
     outputs = trans_stack[2](outputs, seq_len, get_index=0)
     ```

3. 封装模型 **([model.py](./pytorch/model.py))**

   封装的模型不可以脱离项目提供的构造环境运行，均提供参数初始化函数，需要在类实例化后调用。

   * CNN_model(emb_matrix, args, kernel_widths)

     常规CNN模型的封装，支持多种卷积核宽度的同时输入，暂不支持层级结构，模型返回最后的预测结果。

   * RNN_model(emb_matrix, args, mode='classify')

     常规RNN层次模型的封装，支持多层次的分类或序列标注，参数mode可选"classify"/"sequence"，模型返回最后的预测结果。

   * transformer_model(emb_matrix, args)

     常规transformer模型的封装，支持多层transformer的叠加，通过超参数n_layer进行层数控制，模型返回最后的预测结果。

4. 运行模块 **([exec.py](./pytorch/exec.py))**

   * exec(data_dict, args=None, class_name=None)

     基础运行模块，封装运行所需要的三种基本模式：

     (1) train_test(verbose=2)：训练-测试数据的调用函数

     (2) train_itself(verbose=2)：单一训练数据并使用本身进行测试的调用函数

     (3) cross_validation(fold=10, verbose=2)：k折交叉数据的调用函数

   * CNN_classify(data_dict, emb_matrix=None, args=None, kernel_widths=[1, 2, 3], class_name=None)

     **使用CNN分类的执行模块。**

   * RNN_classify(data_dict, emb_matrix=None, args=None, class_name=None)

     **使用RNN分类的执行模块。**

   * RNN_sequence(data_dict, emb_matrix=None, args=None, vote=False, class_name=None)

     **使用RNN序列标注的执行模块。**

   * transformer_classify(data_dict, emb_matrix=None, args=None, class_name=None)

     **使用Transformer分类的执行模块。**

</br>

#### 模型功能及使用

* 分类

  ```python
  from pytorch.base import default_args
  from pytorch.exec import RNN_classify
  
  emb_mat = np.array([...])
  data_dict = {...}
  args = default_args(data_dict)
  class_name = ['support', 'deny', 'query', 'comment']
  nn = RNN_classify(data_dict, emb_mat, args, class_name=class_name)
  nn.cross_validation(fold=10)
  ```

* 序列标注

  ```python
  from pytorch.base import default_args
  from pytorch.exec import RNN_sequence
  
  emb_mat = np.array([...])
  data_dict = {...}
  args = default_args(data_dict)
  class_name = ['support', 'deny', 'query', 'comment']
  nn = RNN_sequence(data_dict, emb_mat, args, vote=False, class_name=class_name)
  nn.train_test()
  ```

* 多次运行同一模型并取Accuracy的平均

  ```python
  args.score_standard = 'Acc'
  nn = CNN_classify(data_dict, emb_mat, args, kernel_widths=[1, 2, 3, 4])
  nn.average_several_run(nn.cross_validation, times=5, fold=5, verbose=2)
  ```

* 网格搜索调参

  ```python
  args.score_standard = 'C1-F'
  nn = transformer_classify(data_dict, emb_mat, args, class_name=class_name)
  params_search = {"l2_reg": [1e-3, 1e-5], "batch_size": [64, 128]}
  nn.grid_search(nn.train_test, params_search=params_search)
  ```

</br>

#### 在GPU服务器上的使用

```python
from dnn.pytorch import base, layer, model, exec
```

所有涉及到的工具包（包括word_vector/predict_analysis/step_print/…）在服务器上也可以直接import。

</br>

#### 模型扩展与重写

- 深度神经网络模型的构建需要继承\<nn.Module\>，建议同时继承基类\<base\>，可以简化参数的使用。在构建时，涉及到已经封装好的网络层级，可以直接调用\<RNN_layer\>、\<self_attention_layer\>等。

  \<RNN_model\>作为一个RNN模型构建的标准示范，扩展和重写的时候可以作为参考。

- 为执行新构造的模型，需要编写一个入口，建议继承\<exec\>，通常，只需要重写内部函数\_run()即可完成运行模块的编写，复杂模型还可能需要重写内部函数\_run_train()/\_run_test()。

- 模型执行模块\<exec\>中，集成了大量的输出规范及控制内容，基础修改输出只需要重写内部函数\_init_display()即可。重写要求提供变量"prf"&"col"&"width"，可以添加到输出列表中的键值包括["Step", "Loss", "Ma-P", "Ma-F", "Ma-F", "Acc", "C0-P", "C0-R", "C0-F", "C1-P", …, "Correct"]。

- [contrib.py](./pytorch/contrib.py)中包含了部分正在实验中的其它模块。

</br>

#### 注意事项

1. 使用Embedding时，应在0位置添加全零向量，以保证在序列补0的情况下，Embedding查询后的向量依然为全零（#不会导致运算错误和结果异常的建议）。
2. 参数space_turbo提供了一种将全部数据直接存入GPU中的加速方法，关闭此功能数据将按batch顺序依次拷贝到GPU中，降低GPU显存消耗但增加程序运行时间，非必要情况不建议关闭space_turbo。
3. 表示层级功能的类应以"\_layer"结尾，表示标准模型的类应以"\_model"结尾，表示模型执行的类应以"\_classify"/"\_sequence"等功能性标注结尾。

