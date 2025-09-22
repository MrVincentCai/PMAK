import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)

import dill
import pandas as pd
import torch
from rdkit import Chem

all_data = pd.read_pickle('../../data/kcat_data/random_final_kcat_dataset.pkl')
print("data_len:", len(all_data))  # 4271
print("文件已加载到data")

all_substrates_smiles = list()
all_products_smiles = list()

for index, data in all_data.iterrows():
    substrates_smiles = list()
    products_smiles = list()
    for substrates_inchi in data['substrates']:
        try:
            mol = Chem.inchi.MolFromInchi(substrates_inchi)
            smiles = Chem.MolToSmiles(mol)
            substrates_smiles.append(smiles)
        except:
            pass
    all_substrates_smiles.append(substrates_smiles)

    for products_inchi in data['products']:
        try:
            mol = Chem.inchi.MolFromInchi(products_inchi)
            smiles = Chem.MolToSmiles(mol)
            products_smiles.append(smiles)
        except:
            pass
    all_products_smiles.append(products_smiles)


all_data["Substrate SMILES"] = all_substrates_smiles
all_data["Product SMILES"] = all_products_smiles
dill.dump(all_data, open('../../data/kcat_data/random_final_kcat_dataset_withSMILES.pkl', 'wb'))


model, tokenizer = get_default_model_and_tokenizer()

rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

Kcat_data = pd.read_pickle('../../data/kcat_data/random_final_kcat_dataset_withSMILES.pkl')
# rxnfp = pd.read_pickle('../../data/kcat_data/rxnfp.pkl')

rxnfp_list = list()
reaction_smiles_list = list()
for index, data in Kcat_data.iterrows():
    substrate_smiles_list = data['Substrate SMILES']  # 
    product_smiles_list = data['Product SMILES']  # 

    substrate_mols = [Chem.MolFromSmiles(smiles) for smiles in substrate_smiles_list]
    product_mols = [Chem.MolFromSmiles(smiles) for smiles in product_smiles_list]

    reaction = rdChemReactions.ChemicalReaction()

    for substrate_mol in substrate_mols:
        reaction.AddReactantTemplate(Chem.RWMol(substrate_mol))
    for product_mol in product_mols:
        reaction.AddProductTemplate(Chem.RWMol(product_mol))

    reaction_smiles = rdChemReactions.ReactionToSmiles(reaction)
    print(reaction_smiles)  # CC(=O)CCC(=O)O.CC(C)(C)C(=O)O>>CC(=O)CC(=O)OCC(=O)CCC(=O)O.CC(=O)CC(=O)OCC(=O)CCC(=O)O
    fp = rxnfp_generator.convert(reaction_smiles)
    rxnfp_list.append(fp)
    reaction_smiles_list.append(reaction_smiles)

rxnfp = pd.DataFrame({'Reaction ID': Kcat_data['Reaction ID'], 'rxnfp': rxnfp_list})
rxnfp['Reaction SMILES'] = reaction_smiles_list
rxnfp.to_pickle("../../data/kcat_data/rxnfp.pkl")