
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from ban import BANLayer



class InteractPre(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1d = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=1)

        
        
        
        

        
        self.W_attention1 = nn.Linear(256, 64)  
        self.W_attention2 = nn.Linear(256, 64)

        self.attention_layer = nn.Linear(64, 64)
        self.protein_attention_layer = nn.Linear(64, 64)
        self.rxnfp_attention_layer = nn.Linear(64, 64)

        
        
        

        self.gate = nn.Linear(128, 64)

        self.drop1 = nn.Dropout(0.75)  
        self.drop2 = nn.Dropout(0.5)  
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def attention_cnn(self, q, k):
        rxnfp_attn = self.rxnfp_attention_layer(q)  
        protein_att = self.protein_attention_layer(k)  
        rxnfp_att_layers = torch.unsqueeze(rxnfp_attn, 1).repeat(1, k.shape[0], 1)  
        protein_att_layers = torch.unsqueeze(protein_att, 0).repeat(q.shape[0], 1, 1)  

        Atten_matrix = self.attention_layer(torch.relu(rxnfp_att_layers + protein_att_layers))  

        rxnfp_atte = torch.mean(Atten_matrix, 1)  
        Protein_atte = torch.mean(Atten_matrix, 0)  
        

        rxnfp_atte = torch.sigmoid(rxnfp_atte)  
        Protein_atte = torch.sigmoid(Protein_atte)  
        self.attn = torch.mean(Protein_atte, dim=1)

        rxnfp = q + q * rxnfp_atte  
        protein = k + k * Protein_atte  

        protein = protein.permute(1, 0).unsqueeze(0)
        protein = F.max_pool1d(protein, kernel_size=protein.shape[-1])
        protein = protein.squeeze(2)
        context_vector = torch.cat((rxnfp, protein), dim=1)

        return context_vector

    def forward(self, reactions, protein):  
        
        

        protein = protein.permute(0, 2, 1)
        protein = torch.relu(self.conv1d(protein))
        protein = protein.permute(0, 2, 1)  

        
        
        
        
        
        
        

        reactions = torch.relu(self.W_attention2(reactions.squeeze(0)))
        protein = torch.relu(self.W_attention1(protein.squeeze(0)))


        aggreX = self.attention_cnn(reactions, protein)
        
        



        self.interact = aggreX

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
