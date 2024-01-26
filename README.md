

![python version](https://img.shields.io/badge/python-v.3.9-blue)
![license](https://img.shields.io/badge/license-MIT-orange)


<h1 id="title">Molbotomy</h1>

[//]: # (![Figure 1]&#40;figures/Fig1.png&#41;)

 
## Description
Molbotomy is a easy-to-use toolkit to split molecular data into a train and test set, either in a balanced way, or by 
creating out-of-distribution splits. 

## Modules
- `Splitter`: Splits molecules into a train and test set
- `MolecularDistanceMatrix`: Computes a pairwise distance matrix of molecules
- `ClusterMolecularDistanceMatrix`: Clusters a distance matrix
- `SpringCleaning`: Clean up a set of molecules
- `Eval`: Evaluate the train and test split and compute some statistics on them.
 
## Usage

**Splitting data:**
```angular2html
from molbotomy import Splitter

smiles = ['NC(=O)c1ccc(N2CCCN(Cc3ccc(F)cc3)CC2)nc1', 'CC(=O)NCC1(C)CCC(c2ccccc2)(N(C)C)CC1', ...]

S = Splitter(split_method='scaffold', mode='ood')
train_idx, test_idx = S.split(smiles, ratio=0.2)
```

**Cleaning data:**
```angular2html
from molbotomy import SpringCleaning

smiles = ['NC(=O)c1ccc(N2CCCN(Cc3ccc(F)cc3)CC2)nc1', 'CC(=O)NCC1(C)CCC(c2ccccc2)(N(C)C)CC1', ...]

C = SprintCleaning(...)
smiles_clean = C.clean(smiles)

C.summary()

> bla bla bla
```

**Evaluating splits:**
```angular2html
from molbotomy import Evaluator

train_smiles = ['NC(=O)c1ccc(N2CCCN(Cc3ccc(F)cc3)CC2)nc1', 'CC(=O)NCC1(C)CCC(c2ccccc2)(N(C)C)CC1', ...]
test_smiles = ['CC(C)C[C@]1(c2cccc(O)c2)CCN(CC2CC2)C1', 'NC(=O)c1ccc(Oc2ccc(CN3CCCC3c3ccccc3)cc2)nc1', ...]

E = Evaluator.eval(train_smiles, test_smiles)
E.summary()
```

 
## Requirements
Install dependencies from the provided env.yaml file.

```conda env create -f env.yaml```

This codebase uses Python 3.9 and depends on:
- [RDKit](https://www.rdkit.org/) (2023.3.2)
- [Scikit-learn](https://scikit-learn.org/) (1.3.0)

<!-- License-->
<h2 id="License">License</h2>

All code is under MIT license.
