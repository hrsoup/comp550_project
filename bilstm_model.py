import torch.nn as nn

class BILSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        super(BILSTM, self).__init__()
        self.linear1 = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.linear2 = nn.Linear(2*hidden_size, out_size)

    def forward(self, x):
        self.bilstm.flatten_parameters()
        emb_out = self.linear1(x)  # linear 1
        bilstm_out, _ = self.bilstm(emb_out) # bilstm
        scores = self.linear2(bilstm_out)  # linear 2

        return scores