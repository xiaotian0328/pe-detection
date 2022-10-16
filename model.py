import torch
from torch import nn
from torch.nn.utils import weight_norm


# https://github.com/GuanshuoXu/RSNA-STR-Pulmonary-Embolism-Detection/blob/main/trainval/2nd_level/seresnext50_128.py
# line 17-57
# modified
class Attention(nn.Module):
    def __init__(self, feature_size, seq_len, bias=True):
        super().__init__()
        self.feature_size = feature_size
        self.seq_len = seq_len
        self.bias = bias

        weight = torch.zeros(feature_size, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(seq_len))

    def forward(self, x, mask):
        # x: (batch, seq_len, feature_size)
        # mask: (batch, seq_len)
        e_ij = torch.mm(x.contiguous().view(-1, self.feature_size),  # (batch*seq_len, feature_size)
                        self.weight).view(-1, self.seq_len)  # e_ij: (batch, seq_len)
        if self.bias:
            e_ij = e_ij + self.b
        e_ij = torch.tanh(e_ij)
        alpha = torch.exp(e_ij)  # (batch, seq_len)
        alpha = alpha * mask
        alpha = alpha / torch.sum(alpha, 1, keepdim=True) + 1e-10
        weighted_sequence = x * torch.unsqueeze(alpha, -1)  # (batch, seq_len, feature_size)
        return torch.sum(weighted_sequence, 1), alpha  # (batch, feature_size), (batch, seq_len)


# According to the paper: Self-Attention Temporal Convolutional Network for
#                         Long-Term Daily Living Activity Detection
class SelfAttention(nn.Module):
    def __init__(self, feature_size, seq_len, dropout):
        super().__init__()
        self.feature_size = feature_size
        self.seq_len = seq_len
        
        self.W_q = nn.Linear(seq_len, 1)
        self.conv_k = weight_norm(nn.Conv1d(feature_size, feature_size, kernel_size=1, stride=1))
        self.conv_v = weight_norm(nn.Conv1d(feature_size, feature_size, kernel_size=1, stride=1))
        
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.dropout2d = nn.Dropout2d(dropout)

    def forward(self, x, mask, gamma=1.0):
        # x: (batch, seq_len, feature_size)
        # mask: (batch, seq_len)
        x = x.transpose(1, 2)  # (batch_size, feature_size, seq_len)
        mask = mask.unsqueeze(2)  # (batch_size, seq_len, 1)
        query = self.W_q(self.dropout(x))  # (batch_size, featurez_size, 1)
        key = self.conv_k(self.dropout(x))  # (batch_size, featurez_size, seq_len)
        value = self.conv_v(self.dropout(x))  # (batch_size, featurez_size, seq_len)

        score = torch.bmm(key.transpose(1, 2), query)  # (batch_size, seq_len, 1)
        alpha = self.softmax(score) * mask  # (batch_size, seq_len, 1)

        output = torch.bmm(value, alpha)  # (batch_size, featurez_size, 1)
        return output.squeeze(2), alpha.squeeze(2)  # (batch, feature_size), (batch, seq_len)


# https://github.com/GuanshuoXu/RSNA-STR-Pulmonary-Embolism-Detection/blob/main/trainval/2nd_level/seresnext50_192.py
# line 177-184
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, seq_len, 1, feature_size)
        x = x.permute(0, 3, 2, 1)  # (N, feature_size, 1, seq_len)
        x = super(SpatialDropout, self).forward(x)  # (N, feature_size, 1, seq_len), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, seq_len, 1, feature_size)
        x = x.squeeze(2)  # (N, seq_len, feature_size)
        return x


# copied from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.batchnorm1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.batchnorm2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.batchnorm1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.batchnorm2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# +
class PEFeatureSequentialNet(nn.Module):
    def __init__(self, 
                 seq_model='TCN',  # 'TCN' or 'GRU'
                 input_size=512,  
                 hidden_size=64, 
                 levels=2,
                 kernel_size=3, 
                 seq_len=40, 
                 dropout=0.5, 
                 bidirectional=True, 
                 diff=True, 
                 maxpool=True, 
                 batchnorm=False):
        super().__init__()

        self.label_list = [
            'pe_present_on_image',
            'negative_exam_for_pe',
            'indeterminate',
            'chronic_pe',
            'acute_and_chronic_pe',
            'central_pe',
            'leftsided_pe',
            'rightsided_pe',
            'rv_lv_ratio_gte_1',
            'rv_lv_ratio_lt_1'
        ]

        if diff:
            input_size = input_size * 3

        ratio = 1

        self.seq_model = seq_model
        if seq_model == 'TCN':
            ## Temporal CNN ##
            self.tcn = TemporalConvNet(num_inputs=input_size, 
                                       num_channels=[hidden_size]*levels, 
                                       kernel_size=kernel_size, 
                                       dropout=dropout)
        elif seq_model == 'GRU':
            ## Bidirectional GRU ##
            self.gru = nn.GRU(input_size=input_size, 
                              hidden_size=hidden_size, 
                              batch_first=True, 
                              dropout=dropout, 
                              bidirectional=bidirectional)
            if bidirectional:
                ratio = ratio * 2

        # self.spatialdropout = SpatialDropout(p=0.2)
        self.dropout = nn.Dropout(dropout)

        self.linear_pe = nn.Linear(hidden_size * ratio, 1)

        # attention mechanism on outputs of sequential model
        self.attentions = nn.ModuleDict({label: Attention(hidden_size * ratio, seq_len) for label in self.label_list})
#         self.attention = Attention(hidden_size * ratio, seq_len)

        self.maxpool = maxpool
        if maxpool:
            ratio = ratio * 2

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn = nn.BatchNorm1d(hidden_size * ratio)

        # last layer linear models - output logits
        self.linears = nn.ModuleDict({label: nn.Linear(hidden_size * ratio, 1) for label in self.label_list})
#         self.linear_negative_exam_for_pe = nn.Linear(hidden_size * ratio, 1)
#         self.linear_indeterminate = nn.Linear(hidden_size * ratio, 1)
#         self.linear_chronic_pe = nn.Linear(hidden_size * ratio, 1)
#         self.linear_acute_and_chronic_pe = nn.Linear(hidden_size * ratio, 1)
#         self.linear_central_pe = nn.Linear(hidden_size * ratio, 1)
#         self.linear_leftsided_pe = nn.Linear(hidden_size * ratio, 1)
#         self.linear_rightsided_pe = nn.Linear(hidden_size * ratio, 1)
#         self.linear_rv_lv_ratio_gte_1 = nn.Linear(hidden_size * ratio, 1)
#         self.linear_rv_lv_ratio_lt_1 = nn.Linear(hidden_size * ratio, 1)

    def forward(self, x, mask):
        # x: (batch_size, seq_len, feature_size)
        if self.seq_model == 'TCN':
            x = x.transpose(1, 2)  # (batch_size, feature_size, seq_len)
            h = self.tcn(x)  # (batch_size, hiddens_size, seq_len)
            h = h.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        elif self.seq_model == 'GRU':
            h, _ = self.gru(x)  # (batch_size, seq_len, 2 x hiddens_size)

        logits = {label: None for label in self.label_list}

        logits['pe_present_on_image'] = self.linear_pe(self.dropout(h))  # (batch_size, seq_len, 1)

#         # attention mechanism - (batch_size, hidden_size)
#         att_pool, self.alpha = self.attention(h, mask)

#         # max pooling - (batch_size, hidden_size)
#         if self.maxpool:
#             max_pool, _ = torch.max(h, 1)
#             # concatenating maxpooling and attention pooling
#             embedding = torch.cat((max_pool, att_pool), 1)  # (batch_size, 2 x hidden_size)
#         else:
#             embedding = att_pool

#         if self.batchnorm:
#             embedding = self.bn(embedding)

        self.alphas = {label: None for label in self.label_list[1:]}
        for label in self.label_list[1:]:
            att_pool, self.alphas[label] = self.attentions[label](h, mask)
            if self.maxpool:
                max_pool, _ = torch.max(h, 1)
                embedding = torch.cat((max_pool, att_pool), 1)
            else:
                embedding = att_pool
            if self.batchnorm:
                embedding = self.bn(embedding)
            logits[label] = self.linears[label](self.dropout(embedding))

#         logits_negative_exam_for_pe = self.linear_negative_exam_for_pe(self.dropout(embedding))
#         logits_indeterminate = self.linear_indeterminate(self.dropout(embedding))
#         logits_chronic_pe = self.linear_chronic_pe(self.dropout(embedding))
#         logits_acute_and_chronic_pe = self.linear_acute_and_chronic_pe(self.dropout(embedding))
#         logits_central_pe = self.linear_central_pe(self.dropout(embedding))
#         logits_leftsided_pe = self.linear_leftsided_pe(self.dropout(embedding))
#         logits_rightsided_pe = self.linear_rightsided_pe(self.dropout(embedding))
#         logits_rv_lv_ratio_gte_1 = self.linear_rv_lv_ratio_gte_1(self.dropout(embedding))
#         logits_rv_lv_ratio_lt_1 = self.linear_rv_lv_ratio_lt_1(self.dropout(embedding))

        return logits
#                logits_pe, \
#                logits_negative_exam_for_pe, \
#                logits_indeterminate, \
#                logits_chronic_pe, \
#                logits_acute_and_chronic_pe, \
#                logits_central_pe, \
#                logits_leftsided_pe, \
#                logits_rightsided_pe, \
#                logits_rv_lv_ratio_gte_1, \
#                logits_rv_lv_ratio_lt_1

    def get_attention_weights(self):
        return self.alphas
