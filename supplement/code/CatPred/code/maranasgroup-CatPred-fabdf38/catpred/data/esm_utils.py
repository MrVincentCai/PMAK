import torch
import os
import re
from pathlib import Path
from functools import partial
import esm
from torch.nn.utils.rnn import pad_sequence
from .cache_utils import cache_fn, run_once, md5_hash_fn


def exists(val):
    return val is not None

def map_values(fn, dictionary):
    return {k: fn(v) for k, v in dictionary.items()}

def to_device(t, *, device):
    return t.to(device)

def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

PROTEIN_EMBED_USE_CPU = os.getenv('PROTEIN_EMBED_USE_CPU', None) is not None

if PROTEIN_EMBED_USE_CPU:
    print('calculating protein embed only on cpu')



GLOBAL_VARIABLES = {
    'model': None,
    'tokenizer': None
}


import ipdb

def calc_protein_representations_with_subunits(proteins, get_repr_fn, *, device):
    representations = []
    for subunits in proteins:
        subunits = cast_tuple(subunits)
        try:
            subunits_representations = list(map(get_repr_fn, subunits))
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, retrying batch')
                torch.cuda.empty_cache()
                
                subunits_representations = list(map(get_repr_fn, subunits))
            else:
                raise e
        subunits_representations = list(map(partial(to_device, device = device), subunits_representations))
        subunits_representations = torch.cat(subunits_representations, dim = 0)
        representations.append(subunits_representations)

    lengths = [seq_repr.shape[0] for seq_repr in representations]
    masks = torch.arange(max(lengths), device = device)[None, :] <  torch.tensor(lengths, device = device)[:, None]
    
    padded_representations = pad_sequence(representations, batch_first = True)

    return padded_representations.to(device), masks.to(device)



ESM_MAX_LENGTH = 2048
ESM_EMBED_DIM = 1280

INT_TO_AA_STR_MAP = {
    0: '<cls>',
    1: '<pad>',
    2: '<eos>',
    3: '<unk>',
    4: 'L',
    5: 'A',
    6: 'G',
    7: 'V',
    8: 'S',
    9: 'E',
    10: 'R',
    11: 'T',
    12: 'I',
    13: 'D',
    14: 'P',
    15: 'K',
    16: 'Q',
    17: 'N',
    18: 'F',
    19: 'Y',
    20: 'M',
    21: 'H',
    22: 'W',
    23: 'C',
    24: 'X',
    25: 'B',
    26: 'U',
    27: 'Z',
    28: 'O',
    29: '.',
    30: '-',
    31: '<null_1>',
    32: '<mask>'
}
AA_STR_TO_INT_MAP = {v:k for k,v in INT_TO_AA_STR_MAP.items()}

import ipdb

def tensor_to_aa_str(t):
    str_seqs = []
    
    for int_seq in t.unbind(dim = 0):
        str_seq = list(map(lambda t: INT_TO_AA_STR_MAP[t] if t != 20 else '', int_seq.tolist()))
        str_seqs.append(''.join(str_seq))
    return str_seqs

@run_once('init_esm')
def init_esm():
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    if not PROTEIN_EMBED_USE_CPU:
        model = model.cuda()

    GLOBAL_VARIABLES['model'] = (model, batch_converter)

@run_once('init_esm_if')
def init_esm_if():
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    batch_converter = esm_if.util.CoordBatchConverter(alphabet, 2048)

    if not PROTEIN_EMBED_USE_CPU:
        model = model.cuda()

    GLOBAL_VARIABLES['esmif_model'] = (model, batch_converter)

def get_single_esm_repr(protein_str):
    init_esm()
    model, batch_converter = GLOBAL_VARIABLES['model']

    data = [('protein', protein_str)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    if batch_tokens.shape[1] > ESM_MAX_LENGTH:
        print(f'warning max length protein esm')

    batch_tokens = batch_tokens[:, :ESM_MAX_LENGTH]

    if not PROTEIN_EMBED_USE_CPU:
        batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])

    token_representations = results['representations'][33]
    representation = token_representations[0][1 : len(protein_str) + 1]
    return representation

def get_esm_repr(proteins, name, device):
    if isinstance(proteins, torch.Tensor):
        proteins = tensor_to_aa_str(proteins)
    
    get_protein_repr_fn = cache_fn(get_single_esm_repr, path = 'esm/proteins', name = name)

    return calc_protein_representations_with_subunits([proteins], get_protein_repr_fn, device = device)

def get_coords(pdbpath):
    
    
    addpath = '/home/ubuntu/CatPred-DB/CatPred-DB/'
    coords = esm_if.util.load_coords(addpath+pdbpath, 'A')
    return coords
    
def get_esm_tokens(protein_str, device):
    if isinstance(protein_str, torch.Tensor):
        proteins = tensor_to_aa_str(proteins)
    
    init_esm()
    model, batch_converter = GLOBAL_VARIABLES['model']

    data = [('protein', protein_str)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    if batch_tokens.shape[1] > ESM_MAX_LENGTH:
        print(f'warning max length protein esm')

    batch_tokens = batch_tokens[:, :ESM_MAX_LENGTH]

    if device!='cpu':
        batch_tokens = batch_tokens.cuda()

    return batch_tokens



PROTEIN_REPR_CONFIG = {
    'esm': {
        'dim': ESM_EMBED_DIM,
        'fn': get_esm_repr,
        'tokenizer': get_esm_tokens,
    }
}

def get_protein_embedder(name):
    allowed_protein_embedders = list(PROTEIN_REPR_CONFIG.keys())
    assert name in allowed_protein_embedders, f"must be one of {', '.join(allowed_protein_embedders)}"

    config = PROTEIN_REPR_CONFIG[name]
    return config
