import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Model(torch.nn.Module):
    def __init__(self, n_feature, n_class, labels101to20=None):
        super(Model, self).__init__()
        self.labels20 = labels101to20
        self.n_class = n_class
        n_featureby2 = int(n_feature/2)
        # FC layers for the 2 streams
        self.fc_f = nn.Linear(n_featureby2, n_featureby2)
        self.fc1_f = nn.Linear(n_featureby2, n_featureby2)
        self.fc_r = nn.Linear(n_featureby2, n_featureby2)
        self.fc1_r = nn.Linear(n_featureby2, n_featureby2)
        self.classifier_f = nn.Linear(n_featureby2, n_class)
        self.classifier_r = nn.Linear(n_featureby2, n_class)
        if n_class == 100:
            # temporal conv for activitynet
            self.conv = nn.Conv1d(n_class, n_class, kernel_size=13, stride=1, padding=12, dilation=2, bias=False, groups=n_class)
        
        self.apply(weights_init)
        # Params for multipliers of TCams for the 2 streams
        self.mul_r = nn.Parameter(data=torch.Tensor(n_class).float().fill_(1))
        self.mul_f = nn.Parameter(data=torch.Tensor(n_class).float().fill_(1))
        self.dropout_f = nn.Dropout(0.7)
        self.dropout_r = nn.Dropout(0.7)
        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=1)


    def forward(self, inputs, device='cpu', is_training=True, seq_len=None):
        #inputs - batch x seq_len x featSize
        base_x_f = inputs[:,:,1024:]
        base_x_r = inputs[:,:,:1024]
        x_f = self.relu(self.fc_f(base_x_f))
        x_r = self.relu(self.fc_r(base_x_r))
        x_f = self.relu(self.fc1_f(x_f))
        x_r = self.relu(self.fc1_r(x_r))

        if is_training:
            x_f = self.dropout_f(x_f)
            x_r = self.dropout_r(x_r)
        
        cls_x_f = self.classifier_f(x_f)
        cls_x_r = self.classifier_r(x_r)
        
        tcam = cls_x_r * self.mul_r + cls_x_f * self.mul_f
        ### Add temporal conv for activity-net
        if self.n_class==100:  
            tcam = self.relu(self.conv(tcam.permute([0,2,1]))).permute([0,2,1])

        if seq_len is not None:
            atn = torch.zeros(0).to(device)
            mx_len = seq_len.max()
            for b in range(inputs.size(0)):
                if seq_len[b] == mx_len:
                    atnb1 = self.softmaxd1(tcam[[b],:seq_len[b]])
                else:
                    # attention over valid segments
                    atnb1 = torch.cat([self.softmaxd1(tcam[[b],:seq_len[b]]), tcam[[b],seq_len[b]:]], dim=1)
                atn = torch.cat([atn, atnb1], dim=0)
            # attention-weighted TCAM for count
            if self.labels20 is not None:
                count_feat = tcam[:,:,self.labels20] * atn[:,:,self.labels20]
            else:
                count_feat = tcam * atn
                
        return x_f, cls_x_f, x_r, cls_x_r, tcam, count_feat
