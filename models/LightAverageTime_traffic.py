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
        self.group1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 249, 250, 251, 252, 253, 254, 255, 256, 257, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 286, 287, 288, 289, 290, 291, 292, 293, 294, 296, 297, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 446, 447, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 719, 720, 721, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 856, 857, 858, 859, 860, 861], [285, 607]]
        self.group2 = [45, 56, 248, 258, 295, 298, 445, 448, 569, 718, 722, 723, 737, 762, 790, 855]
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

