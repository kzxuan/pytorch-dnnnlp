## PyTorch - Deep Neural Network - Natural Language Processing

Version 1.0 by KzXuan

**Contains CNN, RNN and Transformer layers and models implemented by pytorch for classification tasks in NLP.**

* Newly designed modules.
* Reduce usage complexity.
* Use `mask` as the sequence length identifier.
* Multi-GPU parallel for grid search.

Coming soon: new sequence labeling support.

<br>

### Introduction

* Dependecies

  python >= 3.5 & pytorch >= 1.2.0

* [API Documents](./docs.md)

  * [Layer](./docs.md#Layer) - [layer.py](./dnnnlp/layer.py)
  * [Model](./docs.md#Model) -  [model.py](./dnnnlp/model.py)
  * [Execution](./docs.md#Execution) - [exec.py](./dnnnlp/exec.py)
  * [Utility](./docs.md#Utility) - [utils.py](./dnnnlp/utils.py)

* Hyperparameters

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

* Classification.

  ````python
  from dnnnlp.model import RNNModel
  from dnnnlp.exec import default_args, Classify

  emb_mat = np.array(...)
  train_x = np.array((800, 50, 300))
  train_y = np.array((800,))
  train_mask = np.array((800, 50))
  test_x = np.array((200, 50, 300))
  test_y = np.array((200,))
  test_mask = np.array((200, 50))

  args = default_args()

  model = RNNModel(args, emb_mat, bi_direction=False, rnn_type='GRU', use_attention=True)

  nn = Classify(model, args, train_x, train_y, train_mask, test_x, test_y, test_mask)
  evals = nn.train_test(device_id=0)
  ````

* Run several times and get the average score.

  ````python
  from dnnnlp.model import CNNModel
  from dnnnlp.exec import default_args, Classify, average_several_run

  emb_mat = np.array(...)
  train_x = np.array((1000, 50, 300))
  train_y = np.array((1000,))
  train_mask = np.array((1000, 50))

  args = default_args()

  model = CNNModel(args, emb_mat, kernel_widths=[2, 3, 4])

  nn = Classify(model, args, train_x, train_y, train_mask)
  avg_evals = average_several_run(nn.cross_validation, args, n_times=8, n_paral=4, fold=5)
  ````

* Parameters' grid search.

  ````python
  from dnnnlp.model import TransformerModel
  from dnnnlp.exec import default_args, Classify, grid_search

  emb_mat = np.array(...)
  train_x = np.array((800, 50, 300))
  train_y = np.array((800,))
  train_mask = np.array((800, 50))
  test_x = np.array((200, 50, 300))
  test_y = np.array((200,))
  test_mask = np.array((200, 50))

  args = default_args()

  model = TransformerModel(args, n_layer=12, n_head=8)

  nn = Classify(model, args, train_x, train_y, train_mask, test_x, test_y, test_mask)
  params_search = {'learning_rate': [0.1, 0.01], 'n_hidden': [50, 100]}
  max_evals = grid_search(nn, nn.train_test, args, params_search)
  ````

<br>

### History



[Top](#PyTorch%20-%20Deep%20Neural%20Network%20-%20Natural%20Language%20Processing)
