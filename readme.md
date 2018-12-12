## PyTorch简单RNN模型

Version 0.9 by KzXuan

**包含了PyTorch实现的简单RNN模型（LSTM和GRU）用于NLP领域的分类及序列标注任务。**

相比TensorFlow的静态图模型，PyTorch拥有更为原生的Python语言写法，默认支持动态建图。

在同数据集同模型同参数的设置下，第一份小数据进行训练+测试用时PyTorch 60s VS TensorFlow 86s；第二份大数据进行十折交叉用时PyTorch 6m26s VS TensorFlow 14m34s，两者的测试准确率结果几乎完全相同，同时PyTorch所占用的GPU资源会更小。



#### 模型说明

环境：

* Python 3.5/3.6/3.7
* PyTorch 0.4.1/1.0.0

超参数说明：

- cuda_enable [bool]	是否使用GPU加速
- GRU_enable [bool]	使用GRU或LSTM
- bi_direction [bool]	双向LSTM/GRU
- n_layer [int]			每个层次的LSTM/GRU层数
- use_attention [bool]	是否使用注意力机制（默认在每一层次的LSTM/GRU上添加）
- emb_type [str]		使用None/'const'/'variable'/'random'表示Embedding模式
- emb_dim [int]		Embedding维度（输入为特征时表示特征的维度）
- n_class [int]			分类的目标类数
- n_hierarchy [int]		RNN模型的层次数
- n_hidden [int]		LSTM/GRU的隐层节点数
- learning_rate [float]	学习率
- l2_reg [float]			L2正则
- batch_size [int]		批量大小
- iter_times [int]		迭代次数
- display_step [int]		迭代过程中显示输出的间隔迭代次数
- drop_prob [float]		Dropout比例
- score_standard [str]	使用'P'/'R'/'F'/'Acc'设定模型评判标准

数据要求：

- **构建data_dict并送入模型，data_dict为数据字典**，包含：
  - 'x' [np.array]		训练集输入数据
  - 'y' [np.array]		训练集标签
  - 'len' [list]		训练集序列长度，列表中元素皆为np.array，从前往后表示模型从下到上每一个序列层级的序列长度
  - 'tx' [np.array]	测试集输入数据，可选
  - 'ty' [np.array]	测试集标签，可选
  - 'tlen' [list]		测试集序列长度，可选

类及函数说明：

* default_args(data_dict=None)：

  接受data_dict作为参数，初始化所有超参数，并返回参数集。所有参数支持在命令行内直接赋值，或在得到返回值后修改。

  **在使用此函数获得默认参数集后，大部分参数将不需要再进行手动修改。**

* base(args)：

  基类，接受args参数集，初始化模型参数，并包含部分基本函数，集成多次运行取平均、参数网格搜索等功能。

* self_attention_model(n_hidden)：

  自注意力机制模型，接受隐层节点数n_hidden参数。

* LSTM_model(input_size, n_hidden, n_layer, drop_prob, bi_direction, GRU_enable=False, use_attention=False)：

  封装好的LSTM/GRU模型，可以独立运行，支持单/双向及注意力机制。

  调用时需要传入一个三维的inputs和一个一维的length来保证模型的正常运行。

  调用时可选择输出模式"all"/"last"/"att"来分别得到最后一层的全部隐层输出，或最后一层的最后一个时间步的输出，或Attention后的输出。

* RNN_model(emb_matrix, args, model='classify')：

  常规RNN层次模型的封装，支持多层次的分类或序列标注，参数model可选"classify"/"sequence"，模型返回最后的预测结果。

* RNN_classify(data_dict, emb_matrix=None, args=None, class_name=None)：

  **RNN分类模型的入口，使用RNN分类的导入类。**

  * train_test(verbose=2)：训练-测试数据的调用函数
  * train_itself(verbose=2)：单一训练数据并使用本身进行测试的调用函数
  * cross_validation(fold=10, verbose=2)：k折交叉数据的调用函数

* RNN_sequence(data_dict, emb_matrix=None, args=None, vote=False, class_name=None)：

  **RNN序列标注模型的入口，使用RNN序列标注的导入类。**

  可调用函数同RNN_classify。



#### 模型功能及使用

* 分类

  ```python
  from model import default_args, RNN_classify
  
  emb_mat = np.array([...])
  data_dict = {...}
  args = default_args(data_dict)
  class_name = ['support', 'deny', 'query', 'comment']
  nn = RNN_classify(data_dict, emb_mat, args, class_name=class_name)
  nn.cross_validation(fold=10)
  ```

* 序列标注

  ```python
  from model import default_args, RNN_sequence
  
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
  nn = RNN_classify(data_dict, emb_mat, args, class_name=class_name)
  nn.average_several_run(nn.cross_validation, times=5, fold=5, verbose=2)
  ```

* 网格搜索调参

  ```python
  args.score_standard = 'F'
  nn = RNN_classify(data_dict, emb_mat, args, class_name=class_name)
  params_search = {"l2_reg": [1e-3, 1e-5], "batch_size": [64, 128]}
  nn.grid_search(nn.train_test, params_search=params_search)
  ```



#### 模型扩展与重写

* 深度神经网络的模型的构建需要继承\<nn.Module\>，建议同时继承基类\<base\>，可以简化参数的使用。在构建时，涉及到LSTM/GRU或者Attention的部分，可以直接调用\<LSTM_model\>和\<self_attention_model\>。

  \<RNN_model\>作为一个RNN模型构建的标准示范，扩展和重写的时候可以作为参考。

* 为执行新构造的模型，需要编写一个入口，建议继承\<RNN_classify\>，通常，只需要重写内部函数\_run()即可完成运行模块的编写，复杂模型还可能需要重写内部函数\_run_train()/\_run_test()。
* 模型执行模块（包括\<RNN_classify>/\<RNN_sequence>）中，集成了大量的输出规范及控制内容，基础修改输出只需要重写内部函数\_init_display()即可。可以添加到输出列表中的键值包括["Step", "Loss", "Ma-P", "Ma-F", "Ma-F", "Acc", "C0-P", "C0-R", "C0-F", "C1-P", …, "Correct"]。



#### 在GPU服务器上的使用

```python
from deep_neural.pytorch import default_args, RNN_classify, RNN_sequence
```

所有涉及到的工具包（包括word_vector/predict_analysis/step_print/…）在服务器上也可以直接import。



#### 注意事项

1. 使用Embedding时，应在0位置添加全零向量，以保证在序列补0的情况下，Embedding查询后的向量依然为全零。（#不会导致运算错误和结果异常的建议）

