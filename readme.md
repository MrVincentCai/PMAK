# Description
This repository contains the code and datasets to reproduce the results and figures and to train the models from our paper "PMAK: Enhancing kcat Prediction Through Residue-Aware Attention Mechanism and Pre-Trained Representations".
# Introduction of PMAK
The turnover number ($k_{cat}$) is a critical parameter in enzyme kinetics, representing the maximum number of substrate molecules an enzyme can convert to product per unit time under optimal conditions. While $k_{cat}$ reflects an enzyme's catalytic efficiency and plays a central role in understanding enzyme activity, existing computational models for predicting this parameter often face significant limitations. Most models rely solely on substrate information, overlooking the integral role of the reaction itself in predicting $k_{cat}$. The only model that attempts to incorporate reaction information, TurNuP, still falls short in both predictive accuracy and interpretability. To address these challenges, we propose PMAK, a deep learning model that leverages pre-trained representation learning and a residue-aware attention mechanism. By incorporating both enzyme sequences and reaction details, PMAK generates robust representations, effectively capturing the complex interplay between enzyme and reaction, and offering superior accuracy and generalization compared to existing methods. Our approach not only enhances the prediction of $k_{cat}$ values but also provides valuable insights into the key features driving enzyme catalysis, making it a more interpretable solution for enzymatic reaction prediction.
# Model Framework
![model.png](./images/model.png)

# Usage
1.Split the raw dataset

```python split_data.py```

2.Generate enzyme embedding by Pro-T5-XL

``python get_prot5.py``

3.Generate reaction embedding by RXNFP

```python get_rxnfp.py```

4.Add embeddings of enzyme and reaction to dataframe

```python Add_representation.py```

5.Train the PMAK model

```python train_kcat.py```

6.The codes and datasets used for comparative experiments on CatPred-DB and TurNuP-DB, as well as those employed for the mutation experiments, are provided in the 'supplement' directory.