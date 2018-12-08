## PyTorch简单RNN模型

**包含了PyTorch实现的简单RNN模型（LSTM和GRU）用于NLP领域的分类及序列标注任务。**

相比TensorFlow，PyTorch拥有更为原生的Python语言写法，直接支持动态建图。



环境：

* Python 3.6/3.7
* PyTorch 0.4.1/1.0.0

超参数说明：

* cuda_enable [bool]	是否使用GPU加速
* GRU_enable [bool]	使用GRU或LSTM
* bi_direction [bool]	双向LSTM/GRU
* n_hierarchy [int]		RNN模型的层级数
* n_layer [int]			每个层级的LSTM/GRU层数
* use_attention [bool]	是否使用注意力机制（默认在每一级LSTM/GRU上添加）
* emb_type [str]		使用None/'const'/'variable'/'random'表示Embedding模式
* emb_dim [int]		Embedding维度（输入为特征时表示特征的维度）
* n_class [int]			分类的目标类数
* n_hidden [int]		LSTM/GRU的隐层节点数
* learning_rate [float]	学习率
* l2_reg [float]			L2正则
* batch_size [int]		批量大小
* iter_times [int]		迭代次数
* display_step [int]		迭代过程中显示输出的间隔迭代次数
* drop_prob [float]		Dropout比例
* score_standard [str]	使用'P'/'R'/'F'/'Acc’设定模型评判标准

数据要求：

* 构建data_dict并送入模型，data_dict为数据字典，包含：
  * ‘x’ [np.array]		训练集输入数据（对应'tx’表示测试集的输入数据）
  * ‘y’ [np.array]		训练集标签（对应'ty’表示测试集的标签）
  * ‘len’ [list]			训练集序列长度，列表中每一个元素为np.array类型，从前往后表示模型从下到上每一个序列层级的序列长度（对应'tlen’表示测试集的序列长度）