## PyTorch简单DNN模型

Version 0.11 by KzXuan

**包含了PyTorch实现的简单DNN模型（CNN & RNN）用于NLP领域的分类及序列标注任务。**

相比TensorFlow的静态图模型，PyTorch拥有更为原生的Python语言写法，默认支持动态建图。

自v0.11起，将略微提高PyTorch所占用的GPU显存，但进一步加快模型运行速度（达到之前的版本的2-3倍，达到TensorFlow静态图模型的2-5倍）。在同数据集同模型同参数的设置下，一份较大的数据进行十折交叉用时PyTorch 4m40s VS TensorFlow 20m06s。

</br>

#### 模型说明

* 环境：
  * Python 3.5/3.6/3.7
  * PyTorch 0.4.1/1.0.0

* 超参数说明：

  | 参数名         | 类型  | 默认值 | 说明                                                  |
  | -------------- | ----- | ------ | ----------------------------------------------------- |
  | n_gpu          | int   | 1      | 使用GPU的数量（0表示不使用GPU加速）                   |
  | GRU_enable     | bool  | True   | 使用GRU或LSTM                                         |
  | bi_direction   | bool  | True   | 双向/单向RNN                                          |
  | n_layer        | int   | 1      | 每个层次的RNN层数                                     |
  | use_attention  | bool  | Ture   | 是否使用注意力机制（默认在每一层次的RNN上添加）       |
  | emb_type       | str   | None   | 使用None/'const'/'variable'/'random'表示Embedding模式 |
  | emb_dim        | int   | 300    | Embedding维度（输入为特征时表示特征的维度）           |
  | n_class        | int   | 2      | 分类的目标类数                                        |
  | n_hierarchy    | int   | 1      | RNN模型的层次数                                       |
  | n_hidden       | int   | 50     | LSTM/GRU的隐层节点数，或CNN的输出通道数               |
  | learning_rate  | float | 0.01   | 学习率                                                |
  | l2_reg         | float | 1e-6   | L2正则                                                |
  | batch_size     | int   | 128    | 批量大小                                              |
  | iter_times     | int   | 30     | 迭代次数                                              |
  | display_step   | int   | 2      | 迭代过程中显示输出的间隔迭代次数                      |
  | drop_prob      | float | 0.1    | Dropout比例                                           |
  | score_standard | str   | 'Acc'  | 使用'Ma-F'/…/'C1-R'/'C1-F'/'Acc'等设定模型评判标准    |

* 数据要求：

  **构建data_dict并送入模型，data_dict为数据字典**，其中包含：

  | 键值   | 类型                 | 说明                                                         | 必须值 |
  | ------ | -------------------- | ------------------------------------------------------------ | ------ |
  | 'x'    | np.array             | 训练集输入数据                                               | 是     |
  | 'y'    | np.array             | 训练集标签                                                   | 是     |
  | 'len'  | list [np.array, ...] | 训练集序列长度，从前往后表示模型从下到上每一个层级的序列长度 | 是     |
  | 'tx'   | np.array             | 测试集输入数据                                               | 否     |
  | 'ty'   | np.array             | 测试集标签                                                   | 否     |
  | 'tlen' | list [np.array, ...] | 测试集序列长度，内容同训练集                                 | 否     |

</br>

#### 代码说明

1. 参数和基类 **(base.py)**

   * default_args(data_dict=None)

     接受data_dict作为参数，初始化所有超参数，并返回参数集。所有参数支持在命令行内直接赋值，或在得到返回值后修改。

     **在使用此函数获得默认参数集后，大部分参数将不需要再进行手动修改。**

   * base(args)

     基类，接受args参数集，初始化模型参数，并包含部分基本函数，集成多次运行取平均、参数网格搜索等功能。

2. 封装网络层 **(layer.py)**

   封装的类都可以脱离项目提供的构造环境来单独运行，且均提供参数初始化函数，需要在类实例化后调用。

   * self_attention_layer(n_hidden)

     自注意力机制层，接受隐层节点数n_hidden参数。

   * CNN_layer(input_size, in_channels, out_channels, kernel_width, stride=1)

     封装的CNN层，支持最大池化和平均池化。

     调用时需要传入一个四维的inputs来保证模型的正常运行，若传入的inputs为三维，会自动添加一个第二维，并在第二维上复制in_channels次。若需要使用长度信息，seq_len必须是一维向量。

     调用时可选择输出模式"max"/"mean"/"all"来分别得到最大池化后的输出，平均池化后的输出或原始的全部输出。

   * LSTM_layer(input_size, n_hidden, n_layer, drop_prob, bi_direction, GRU_enable=False)

     封装的LSTM/GRU层，支持单/双向及注意力机制。

     调用时需要传入一个三维的inputs来保证模型的正常运行。若需要使用长度信息，seq_len必须是一维向量。

     调用时可选择输出模式"all"/"last"来分别得到最后一层的全部隐层输出，或最后一层的最后一个时间步的输出。

   * softmax_layer(n_in, n_out)

     简单的Softmax层/全连接层。

3. 封装模型 **(model.py)**

   封装的模型不可以脱离项目提供的构造环境运行，均提供参数初始化函数，需要在类实例化后调用。

   * CNN_model(emb_matrix, args, kernel_widths)

     常规CNN模型的封装，支持多种卷积核宽度的同时输入，暂不支持层级结构，模型返回最后的预测结果。

   * RNN_model(emb_matrix, args, mode='classify')

     常规RNN层次模型的封装，支持多层次的分类或序列标注，参数mode可选"classify"/"sequence"，模型返回最后的预测结果。

4. 运行模块 **(exec.py)**

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
  nn = RNN_classify(data_dict, emb_mat, args, class_name=class_name)
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

- 深度神经网络模型的构建需要继承\<nn.Module\>，建议同时继承基类\<base\>，可以简化参数的使用。在构建时，涉及到已经封装好的网络层级，可以直接调用\<LSTM_layer\>、\<self_attention_layer\>等。

  \<RNN_model\>作为一个RNN模型构建的标准示范，扩展和重写的时候可以作为参考。

- 为执行新构造的模型，需要编写一个入口，建议继承\<exec\>，通常，只需要重写内部函数\_run()即可完成运行模块的编写，复杂模型还可能需要重写内部函数\_run_train()/\_run_test()。

- 模型执行模块\<exec\>中，集成了大量的输出规范及控制内容，基础修改输出只需要重写内部函数\_init_display()即可。重写要求提供变量"prf"&"col"&"width"，可以添加到输出列表中的键值包括["Step", "Loss", "Ma-P", "Ma-F", "Ma-F", "Acc", "C0-P", "C0-R", "C0-F", "C1-P", …, "Correct"]。

- pytorch.contrib中包含了部分已经用PyTorch复现的扩展模型。

</br>

#### 注意事项

1. 使用Embedding时，应在0位置添加全零向量，以保证在序列补0的情况下，Embedding查询后的向量依然为全零（#不会导致运算错误和结果异常的建议）。
2. 表示层级功能的类应以"\_layer"结尾，表示标准模型的类应以"\_model"结尾，表示模型执行的类应以"\_classify"/"\_sequence"等功能性标注结尾。

