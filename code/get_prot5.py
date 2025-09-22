import os
import torch
from transformers import T5EncoderModel, T5Tokenizer
from transformers import AutoTokenizer, EsmModel, EsmTokenizer
import re
import gc

import pandas as pd

import pickle



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def Seq_to_vec(Sequence):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)


    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")

    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            print("seq_emb shape:", seq_emd.shape)
            features.append(seq_emd)

    return features



if __name__ == '__main__':
    # Dataset Load
    datasets_turnup = pd.read_pickle('random_final_kcat_dataset.pkl')

    sequence_turnup = [data['Sequence'] for index, data in datasets_turnup.iterrows()]

    # Feature Extractor

    sequence_input = Seq_to_vec(sequence_turnup)

    with open("Turnup_Protein_vector_dim1024.pkl", "wb") as f:
        pickle.dump(sequence_input, f)
