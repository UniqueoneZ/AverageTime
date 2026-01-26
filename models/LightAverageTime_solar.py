import numpy as np
import torch
import torch.nn as nn
from layers.RevIN import RevIN
from einops import rearrange, repeat, einsum
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(features, 2 * features, bias=False)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(2 * features, features, bias=False)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.activation(x1)
        x1 = self.linear2(x1)
        return x + x1

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.batch_size = configs.batch_size
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.rnn_type = configs.rnn_type
        self.dec_way = configs.dec_way
        self.seg_len = configs.seg_len
        self.channel_id = configs.channel_id
        self.revin = configs.revin
        self.is_training = configs.is_training
        
        #设置Transformer嵌入的层数，这是指每个Transformer模块模型里边的层数
        self.num_layers = configs.num_layers
        
        #分别设置线性层和transformer结构的嵌入个数
        self.num_layers_trans = configs.num_layers_trans
        self.num_layers_linear = configs.num_layers_linear
        
        #分别设置预测和嵌入的dropout
        self.predict_dropout = configs.dropout
        self.emb_dropout = configs.emb_dropout
        
        #定义我们的分组聚类模式
        self.group1 = [[1, 129, 130, 133, 27, 28, 31, 48, 52, 64, 67, 72, 79, 95, 99, 114, 117, 125, 127], [41, 2, 110], [128, 3, 134, 7, 8, 136, 12, 14, 16, 17, 20, 21, 24, 32, 33, 36, 37, 38, 39, 46, 49, 50, 53, 55, 56, 57, 59, 60, 63, 65, 66, 74, 75, 76, 83, 85, 86, 87, 89, 92, 94, 97, 100, 101, 104, 106, 107, 108, 109, 113, 118, 120, 123, 126], [5, 90, 71, 40, 122, 42, 78, 47, 80, 111, 61, 116, 119, 58, 91, 124, 93], [35, 70, 6, 105, 11, 112, 54], [131, 132, 9, 10, 13, 18, 19, 25, 26, 29, 34, 43, 51, 62, 77, 81, 84, 88, 98, 103]]
        self.group2 = [0, 4, 15, 22, 23, 30, 44, 45, 68, 69, 73, 82, 96, 102, 115, 121, 135]
        #对于所有的周期模式以及周期函数都设置独立的预测函数
        if self.channel_id:
            self.predict_layers1 =  nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.seq_len, self.d_model, bias = False),  # 输出层
                    nn.Dropout(self.predict_dropout),
                    nn.SiLU(),
                    nn.Linear(self.d_model, self.pred_len, bias = False)
                ) for _ in range(len(self.group1))
            ])
            self.predict_layers2 =  nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.seq_len, self.d_model, bias = False),  # 输出层
                    nn.Dropout(self.predict_dropout),
                    nn.SiLU(),
                    nn.Linear(self.d_model, self.pred_len, bias = False)
                ) for _ in range(len(self.group2))
            ])
        else:
            self.predict_layers = nn.Sequential(
                    nn.Linear(self.seq_len, self.d_model, bias = False),  # 输出层
                    nn.Dropout(self.predict_dropout),
                    nn.SiLU(),
                    nn.Linear(self.d_model, self.pred_len, bias = False)
                )





        #定义拓展的通道数
        self.channel_number = configs.c_layers

        #定义一个通道层
        if self.channel_id:
            #对于所有的周期模式以及周期函数都设置独立的预测函数
            self.predict_layers_list1 = nn.ModuleList([
                nn.ModuleList([  # 内部再使用 nn.ModuleList
                    nn.Sequential(
                        nn.Linear(self.seq_len, self.d_model, bias = False),  # 输出层
                        nn.Dropout(self.predict_dropout),
                        nn.SiLU(),
                        nn.Linear(self.d_model, self.pred_len, bias = False)
                    ) for _ in range(len(self.group1))
                ]) for i in range(self.num_layers_linear + self.num_layers_trans)
            ])
            self.predict_layers_list2 = nn.ModuleList([
                nn.ModuleList([  # 内部再使用 nn.ModuleList
                    nn.Sequential(
                        nn.Linear(self.seq_len, self.d_model, bias = False),  # 输出层
                        nn.Dropout(self.predict_dropout),
                        nn.SiLU(),
                        nn.Linear(self.d_model, self.pred_len, bias = False)
                    ) for _ in range(len(self.group2))
                ]) for i in range(self.num_layers_linear + self.num_layers_trans)
            ])
        else:
            self.predict_layers_list = nn.ModuleList([ # 内部再使用 nn.ModuleList
                    nn.Sequential(
                        nn.Linear(self.seq_len, self.d_model, bias = False),  # 输出层
                        nn.Dropout(self.predict_dropout),
                        nn.SiLU(),
                        nn.Linear(self.d_model, self.pred_len, bias = False)
                    ) for i in range(self.num_layers_linear + self.num_layers_trans)
                ])


        self.channel_layer1 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.enc_in, 2 * self.enc_in, bias = False),  # 输出层
                nn.Dropout(self.emb_dropout),
                nn.SiLU(),
                nn.Linear(2 * self.enc_in, self.enc_in, bias = False),  # 输出层
            ) for _ in range(self.num_layers_linear)
            ])
        self.channel_layer2 = nn.ModuleList([
            nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.seq_len,  # 输入特征的维度
                nhead=8,  # 多头注意力的头数
                dim_feedforward=self.d_model,  # 前馈神经网络的隐藏层维度
                batch_first=True  # 输入/输出格式为 (batch, seq_len, features)
                ),
                num_layers=self.num_layers  # TransformerEncoder 的层数
            ) for _ in range(self.num_layers_trans)
            ])


        
        if self.revin:
            self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=False)



    def forward(self, x):
        (batch_size,_,_) = x.shape
        # normalize
        if self.revin:
            x = self.revinLayer(x, 'norm').permute(0, 2, 1)
        else:
            seq_last = x[:, -1:, :].detach()
            x = (x - seq_last).permute(0, 2, 1) # b,c,s

        x_clone = []
        for i in range(self.num_layers_linear):
            x1 = x.clone()
            x_clone.append(self.channel_layer1[i](x1.permute(0,2,1)).permute(0,2,1))
            
        for i in range(self.num_layers_trans):            
            x1 = x.clone()
            x_clone.append(self.channel_layer2[i](x1))
        
        #定义原始数据预测出来的结果
        y = torch.zeros(batch_size, self.enc_in, self.pred_len).to(x.device)
        #定义通独通关
        if self.channel_id:
            for i in range(len(self.group1)):
                y[:,self.group1[i],:] = self.predict_layers1[i](x[:,self.group1[i],:])
            for i in range(len(self.group2)):
                y[:,self.group2[i],:] = self.predict_layers2[i](x[:,self.group2[i],:])
        else:
            y = self.predict_layers(x)

        #定义拓展数据预测出来的结果
        y_list = []
        for i in range(self.num_layers_linear + self.num_layers_trans):
            y_list.append(torch.zeros(batch_size, self.enc_in, self.pred_len).to(x.device))
        
        #预测结果
        if self.channel_id:
            for i in range(self.num_layers_linear + self.num_layers_trans):
                for j in range(len(self.group1)):
                    y_list[i][:,self.group1[j],:] = self.predict_layers_list1[i][j](x_clone[i][:,self.group1[j],:])
                for j in range(len(self.group2)):
                    y_list[i][:,self.group2[j],:] = self.predict_layers_list2[i][j](x_clone[i][:,self.group2[j],:])
        else:
            for i in range(self.num_layers_linear + self.num_layers_trans):
                y_list[i] = self.predict_layers_list[i](x_clone[i])
    
        #累加结果
        for i in range(self.num_layers_linear + self.num_layers_trans):
            y += y_list[i]/(self.num_layers_linear + self.num_layers_trans)
        
        y = y/2

        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last


        return y

