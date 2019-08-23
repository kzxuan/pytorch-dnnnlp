## PyTorch - Deep Neural Network - Natural Language Processing

Version 1.0 by KzXuan

**Contains CNN, RNN and Transformer layers and models implemented by pytorch for classification tasks in NLP.**

* Newly designed modules.
* Reduce usage complexity.
* Use `mask` as the sequence length identifier.
* Multi-GPU parallel for grid search.

Coming soon: new sequence labeling support.

<br>

### Dependecies

python 3.5+ & pytorch 1.2.0+

<br>

### Install

```bash
> pip install dnnnlp
```

<br>

### [API Document](./docs.md)

  * [Layer](./docs.md#Layer) - [layer.py](./dnnnlp/layer.py)
  * [Model](./docs.md#Model) -  [model.py](./dnnnlp/model.py)
  * [Execution](./docs.md#Execution) - [exec.py](./dnnnlp/exec.py)
  * [Utility](./docs.md#Utility) - [utils.py](./dnnnlp/utils.py)

<br>

### Hyperparameters

| Name          | Type  | Default     | Description                                                    |
| ------------- | ----- | ----------- | -------------------------------------------------------------- |
| n_gpu         | int   | 1           | The number of GPUs (0 means no GPU acceleration).              |
| space_turbo   | bool  | True        | Accelerate with more GPU memories.                             |
| data_shuffle  | bool  | Ture        | Disrupt data for training.                                     |
| emb_type      | str   | None        | Embedding modes contain None, 'const' or 'variable'.           |
| emb_dim       | int   | 300         | Embedding dimension (or feature dimension).                    |
| n_class       | int   | 2           | Number of target classes.                                      |
| n_hidden      | int   | 50          | Number of hidden nodes, or output channels of CNN.             |
| learning_rate | float | 0.01        | Learning rate.                                                 |
| l2_reg        | float | 1e-6        | L2 regular.                                                    |
| batch_size    | int   | 128         | Number of samples for one batch.                               |
| iter_times    | int   | 30          | Number of iterations.                                          |
| display_step  | int   | 2           | The number of iterations between each output of the result.    |
| drop_prob     | float | 0.1         | Dropout ratio.                                                 |
| eval_metric   | str   | 'accuracy'  | Evaluation metrics contain 'accuracy', 'macro', 'class1', etc. |

<br>

### Usage

```python
# import our modules
from dnnnlp.model import RNNModel
from dnnnlp.exec import default_args, Classify

# load the embedding matrix
emb_mat = np.array((-1, 300))
# load the train data
train_x = np.array((800, 50))
train_y = np.array((800,))
train_mask = np.array((800, 50))
# load the test data
test_x = np.array((200, 50))
test_y = np.array((200,))
test_mask = np.array((200, 50))

# get the default arguments
args = default_args()
# modify part of the arguments
args.space_turbo = False
args.n_hidden = 100
args.batch_size = 32
```

* Classification

```python
# initilize a model
model = RNNModel(args, emb_mat, bi_direction=False, rnn_type='GRU', use_attention=True)
# initilize a classifier
nn = Classify(model, args, train_x, train_y, train_mask, test_x, test_y, test_mask)
# do training and testing
evals = nn.train_test(device_id=0)
```

* Run several times and get the average score.

````python
# initilize a model
model = CNNModel(args, emb_mat, kernel_widths=[2, 3, 4])
# initilize a classifier
nn = Classify(model, args, train_x, train_y, train_mask)
# run the model several times
avg_evals = average_several_run(nn.cross_validation, args, n_times=8, n_paral=4, fold=5)
````

* Parameters' grid search.

````python
# initilize a model
model = TransformerModel(args, n_layer=12, n_head=8)
# initilize a classifier
nn = Classify(model, args, train_x, train_y, train_mask, test_x, test_y, test_mask)
# set searching params
params_search = {'learning_rate': [0.1, 0.01], 'n_hidden': [50, 100]}
# run grid search
max_evals = grid_search(nn, nn.train_test, args, params_search)
````

<br>

### History

**version 1.0**
  * Rename project `dnn` to `dnnnlp`.
  * Remove file `base`, add file `utils`.
  * Optimize and rename `SoftmaxLayer` and `SoftAttentionLayer`.
  * Rewrite and rename `EmbeddingLayer`, `CNNLayer` and `RNNLayer`.
  * Rewrite `MultiheadAttentionLayer`: a packaging attention layer based on `nn.MultiheadAttention`.
  * Rewrite `TransformerLayer`: support new `MultiheadAttentionLayer`.
  * Optimize and rename `CNNModel`, `RNNModel` and `TransformerModel`.
  * Optimize and rename `Classify`: a highly applicable classification execution module.
  * Rewrite `average_several_run` and `grid_search`: support multi-GPU parallel.
  * Support pytorch 1.2.0.

**version 0.12**
  * Update `RNN_layer`: fully support for tanh, LSTM and GRU.
  * Fix errors in some mask operations.
  * Support pytorch 1.1.0.

Old version [0.12.3](https://github.com/NUSTM/pytorch-dnnnlp/tree/8d2d6c4e432076e13020ae54954aa419f3bb9bce).

**version 0.11**
  * Provides an acceleration method by using more GPU memories.
  * Fix the problem of memory consumption caused by abnormal data reading.
  * Add `multi_head_attention_layer`: packaging multi-head attention for Transformer.
  * Add `Transformer_layer` and `Transformer_model`: packaging Transformer layer and model written by ourself.
  * Support data disruption for training.

**version 0.10**
  * Split the code into four files: `base`, `layer`, `model`, `exec`.
  * Add `CNN_layer` and `CNN_model`: packaging CNN layer and model.
  * Support multi-GPU parallel for each model.

**version 0.9**
  * Fix the problem of output format.
  * Fix the statistical errors in cross-validation part of `LSTM_classify`.
  * Rename: `LSTM_model` to `RNN_layer`, `self_attention` to `self_attention_layer`.
  * Add `softmax_layer`: a packaging fully-connected layer.

**version 0.8**
  * Adjust the applicability of functions in `LSTM_classify` to avoid rewriting in `LSTM_sequence`.
  * Optimize the way of parameter transfer.
  * A more complete evaluation mechanism.

**version 0.7**
  * Add `LSTM_sequence`: a sequence labeling module for `LSTM_model`.
  * Fix the nan-value problem in hierarchical classification.
  * Support pytorch 1.0.0.

**version 0.6**
  * Update `LSTM_classify`: support hierarchical classification.
  * The `GRU_model` is merged into the `LSTM_model`.
  * Adapt to CPU operation.

**version 0.5**
  * Split the running part of `LSTM_classify` to reduce the rewrite of custom models.
  * Add control for visual output.
  * Create function `average_several_run`: support to get the average score after several training and testing.
  * Create function `grid_search`: support parameters' grid search.

**version 0.4**
  * Add `GRU_model`: a packaging GRU model based on `nn.GRU`.
  * Support L2 regular.

**version 0.3**
  * Add `self_attention`: provides attention mechanism support.
  * Update `LSTM_classify`: adapts to complex custom models.

**version 0.2**
  * Support mode selection of embedding.
  * Default usage of `nn.Dropout`.
  * Create function `default_args` to provide default hyperparameters.

**version 0.1**
  * Initilization of project `dnn`: based on pytorch 0.4.1.
  * Add `LSTM_model`: a packaging LSTM model based on `nn.LSTM`.
  * Add `LSTM_classify`: a classification module for LSTM model, which supports train-test and corss-validation.
