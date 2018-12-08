## PyTorch简单RNN模型

**包含了PyTorch实现的简单RNN模型（LSTM和GRU）用于NLP领域的分类及序列标注任务。**

相比TensorFlow，PyTorch拥有更为原生的Python语言写法，直接支持动态建图。



环境：

* Python 3.6/3.7
* PyTorch 0.4.1/1.0.0



超参数说明：

* cuda_enable [bool]  是否使用GPU加速
* GRU_enable [bool]  使用GRU或LSTM
* bi_direction [bool]  双向LSTM/GRU
* n_layer [int]  每个层级的LSTM/GRU层数
* use_attention [bool]  是否使用注意力机制（默认在每一级LSTM/GRU上添加）
* emb_type [str]  使用None/'const'/'variable'/'random'表示Embedding模式
* emb_dim [int]: embedding dimension

​        \* n_class [int]: number of object classify classes

​        \* n_hierarchy [int]: number of RNN hierarchies

​        \* n_hidden [int]: number of hidden layer nodes

​        \* learning_rate [float]: learning rate

​        \* l2_reg [float]: L2 regularization parameter

​        \* batch_size [int]: train batch size

​        \* iter_times [int]: iteration times

​        \* display_step [int]: the interval iterations to display

​        \* drop_prob [float]: drop out ratio

​        \* score_standard [str]: use 'P'/'R'/'F'/'Acc'