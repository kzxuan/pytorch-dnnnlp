import torch
import numpy as np
import torch.nn as nn
import word_vector as wv
import easy_function as ef
import torch.utils.data as Data
import torch.nn.functional as F
from dir_set import dir
from copy import deepcopy
from dnn.pytorch import base, layer
from step_print import table_print, percent
from predict_analysis import predict_analysis

""" Load word embedding """
w2v = wv.load_word2vec(dir.W2V_GOOGLE, type='txt')
emb_mat = w2v.get_matrix()
print("- Embedding matrix size:", emb_mat.shape)

""" Load data """
# train data
x = torch.tensor(np.load(dir.TRAIN + "index(4519,30).npy"), dtype=torch.float)
y = torch.tensor(np.load(dir.TRAIN + "y(4519,4).npy"), dtype=torch.int)
l = torch.tensor(np.load(dir.TRAIN + "len(4519,).npy"), dtype=torch.int)
# test data
tx = torch.tensor(np.load(dir.TEST + "index(1049,30).npy"), dtype=torch.float)
ty = torch.tensor(np.load(dir.TEST + "y(1049,4).npy"), dtype=torch.int)
tl = torch.tensor(np.load(dir.TEST + "len(1049,).npy"), dtype=torch.int)

""" Init arguments """
args = base.default_args()
args.emb_type = 'const'
args.emb_dim = w2v.vector_size
args.n_class = 4
args.n_hidden = 50
args.learning_rate = 0.001
args.l2_reg = 0.0
args.batch_size = 128
args.iter_times = 20
args.display_step = 1
args.drop_porb = 0.2

""" Build Model """
class LSTM_model(nn.Module):
    def __init__(self, emb_matrix, args):
        super(LSTM_model, self).__init__()

        # Embedding layer
        self.emb_mat = layer.embedding_layer(emb_mat, 'const')
        # Drop out layer
        self.drop_out = nn.Dropout(args.drop_prob)
        # LSTM layer
        self.lstm = layer.RNN_layer(args.emb_dim, args.n_hidden, args.n_layer,
                                    args.drop_prob, args.bi_direction, mode="LSTM")
        # SoftMax layer
        bi_direction_num = 2 if args.bi_direction else 1
        self.predictor = layer.softmax_layer(bi_direction_num * args.n_hidden, args.n_class)

    def forward(self, inputs, seq_len):
        # Embedding lookup
        inputs = self.emb_mat(inputs.long())
        now_batch_size, max_seq_len, emb_dim = inputs.size()
        # Drop out
        outputs = self.drop_out(inputs)
        # LSTM
        outputs = self.lstm(inputs, seq_len, out_type='last')
        # SoftMax
        pred = self.predictor(outputs)

        return pred


""" Init model """
model = LSTM_model(emb_mat, args)
# Move model to gpu
model.cuda(0)
# Create optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg
)

""" Init data """
train_data = Data.TensorDataset(x, y, l)
train_loader = Data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
test_data = Data.TensorDataset(tx, ty, tl)
test_loader = Data.DataLoader(test_data, shuffle=True, batch_size=args.batch_size)

""" Train and test """
for it in range(1, args.iter_times + 1):
    # Train
    model.train()
    losses = []
    for (_x, _y, _l) in train_loader:
        # Copy data to gpu
        _x, _y, _l = _x.cuda(0), _y.cuda(0), _l.cuda(0)
        pred = model(_x, _l)
        loss = - torch.sum(_y.float() * torch.log(pred)) / torch.sum(_l).float()
        losses.append(loss.cpu().data.numpy())
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test
    model.eval()
    tx, tl = tx.cuda(0), tl.cuda(0)
    pred = model(tx, tl)
    pred = pred.cpu().data.numpy()
    result = predict_analysis(ty.data.numpy(), pred, one_hot=True, simple=True)

    # Print
    print("* Epoch {}, train loss {:.4f}, test accuracy {:.4f}".format(it, sum(losses) / len(losses), result['Acc']))
