import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from ban import BANLayer



class InteractPre(nn.Module):
    def __init__(self):
        super().__init__()
        # 添加卷积层用于将protein降维
        self.conv1d = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=1)
        # self.conv1d_2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.W_attention1 = nn.Linear(256, 64)  # 64
        self.W_attention2 = nn.Linear(256, 64)

        self.attention_layer = nn.Linear(64, 64)
        self.protein_attention_layer = nn.Linear(64, 64)
        self.rxnfp_attention_layer = nn.Linear(64, 64)

        # self.bcn = weight_norm(
        #     BANLayer(v_dim=64, q_dim=64, h_dim=128, h_out=1),
        #     name='h_mat', dim=None)

        # self.gate = nn.Linear(128, 64)

        self.drop1 = nn.Dropout(0.75)  # special drop_rate  # 0.5  0.75
        self.drop2 = nn.Dropout(0.5)  # 0.2  0.5
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    # Neural Attention
    # def attention_cnn(self, x, xs):
    #
    #
    #     weights = torch.tanh(F.linear(x, xs))
    #     ys = torch.t(weights) * xs
    #     ys = xs + ys
    #
    #     protein = ys.permute(1, 0).unsqueeze(0)
    #     protein = F.max_pool1d(protein, kernel_size=protein.shape[-1])
    #     protein = protein.squeeze(2)
    #
    #     out = torch.cat((x, protein), dim=1)
    #
    #     self.attn = weights
    #
    #     return out

    def attention_cnn(self, q, k):
        rxnfp_attn = self.rxnfp_attention_layer(q)  # [1, 64]
        protein_att = self.protein_attention_layer(k)  # [n, 64]
        rxnfp_att_layers = torch.unsqueeze(rxnfp_attn, 1).repeat(1, k.shape[0], 1)  # [1, n, 64]
        protein_att_layers = torch.unsqueeze(protein_att, 0).repeat(q.shape[0], 1, 1)  # [1, n, 64]

        Atten_matrix = self.attention_layer(torch.relu(rxnfp_att_layers + protein_att_layers))  # [1, n, 64]

        rxnfp_atte = torch.mean(Atten_matrix, 1)  # [1, 64]
        Protein_atte = torch.mean(Atten_matrix, 0)  # [n, 64]
        # print("Protein_atte:", Protein_atte.shape)

        rxnfp_atte = torch.sigmoid(rxnfp_atte)  #
        Protein_atte = torch.sigmoid(Protein_atte)  #
        self.attn = torch.mean(Protein_atte, dim=1)

        rxnfp = q + q * rxnfp_atte  # [1, 64]  # * 0.5
        protein = k + k * Protein_atte  # [n, 64]  # * 0.5

        protein = protein.permute(1, 0).unsqueeze(0)
        protein = F.max_pool1d(protein, kernel_size=protein.shape[-1])
        protein = protein.squeeze(2)
        context_vector = torch.cat((rxnfp, protein), dim=1)

        return context_vector

    def forward(self, reactions, protein, use_gate=False):  # , crafted_feature
        # print("reactions shape:", reactions.shape)
        # print("protein shape:", protein.shape)
        protein = protein.permute(0, 2, 1)
        protein = torch.relu(self.conv1d(protein))
        protein = protein.permute(0, 2, 1)  # [batch_size, seq_len, 100]

        reactions = torch.relu(self.W_attention2(reactions.squeeze(0)))
        protein = torch.relu(self.W_attention1(protein.squeeze(0)))


        aggreX = self.attention_cnn(reactions, protein)
        # aggreX, att = self.bcn(reactions.unsqueeze(0), protein.unsqueeze(0))
        # print("aggreX shape:", aggreX.shape)

        self.interact = aggreX

        if use_gate:
            gate = torch.sigmoid(self.gate(aggreX))
            protein = aggreX[:, :64]
            rxnfp = aggreX[:, 64:]
            protein = torch.mul(gate, protein)
            rxnfp = torch.mul((1 - gate), rxnfp)
            aggreX = torch.cat((rxnfp, protein), dim=1)

        output = self.drop1(F.leaky_relu(self.fc1(aggreX)))
        output = self.drop2(F.leaky_relu(self.fc2(output)))
        output = self.fc3(output)

        return output.view(-1)

    @property
    def get_aggreX(self):
        return self.interact

    @property
    def get_attn(self):
        return self.attn