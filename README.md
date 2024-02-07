

![python version](https://img.shields.io/badge/python-v.3.9-blue)
![license](https://img.shields.io/badge/license-MIT-orange)


<h1 id="title">Molbotomy</h1>

[//]: # (![Figure 1]&#40;figures/Fig1.png&#41;)

 
## Description
Molbotomy is a easy-to-use toolkit to clean, process, and split molecular data.

## Functionality
- `SpringCleaning`: Clean up a set of molecules
- `scaffold_split, random_split, stratified_random_split`: Splits molecules into a train and test set
- `check_splits`: Do a sanity check on the train and test split and compute some statistics (and/or distances)
- `MolecularDistanceMatrix`: Computes a pairwise distance matrix of molecules

## Usage

**Cleaning data:**
```angular2html
from molbotomy import SpringCleaning

smiles = ['NC(=O)c1ccc(N2CCCN(Cc3ccc(F)cc3)CC2)nc1', 'CC(=O)NCC1(C)CCC(c2ccccc2)(N(C)C)CC1', ...]

C = SpringCleaning(neutralize=True,
                   check_for_uncommon_atoms=True,
                   desalt=True,
                   remove_solvent=True,
                   sanitize=True,
                   ...)

smiles_clean = C.clean(smiles)

C.summary()

> Parsed 10000 molecules of which 9964 successfully.
  Failed to clean 36 molecules: {'unfamiliar token': 25, 'fragmented SMILES': 11}
```

**Splitting data:**
```angular2html
from molbotomy import scaffold_split, tools

smiles = ['NC(=O)c1ccc(N2CCCN(Cc3ccc(F)cc3)CC2)nc1', 'CC(=O)NCC1(C)CCC(c2ccccc2)(N(C)C)CC1', ...]
mols = tools.smiles_to_mols(smiles, sanitize=True)

train_idx, test_idx = scaffold_split(mols, ratio=0.2)

```

**Evaluating splits:**
```angular2html
from molbotomy import check_splits

train_smiles = ['NC(=O)c1ccc(N2CCCN(Cc3ccc(F)cc3)CC2)nc1', 'CC(=O)NCC1(C)CCC(c2ccccc2)(N(C)C)CC1', ...]
test_smiles = ['CC(C)C[C@]1(c2cccc(O)c2)CCN(CC2CC2)C1', 'NC(=O)c1ccc(Oc2ccc(CN3CCCC3c3ccccc3)cc2)nc1', ...]

check_splits(train_smiles, test_smiles)

> Data leakage:
      Found 0 intersecting SMILES between the train and test set.
      Found 0 intersecting Bemis-Murcko scaffolds between the train and test set.
      Found 0 intersecting stereoisomers between the train and test set.
  Duplicates:
      Found 0 duplicate SMILES in the train set.
      Found 0 duplicate SMILES in the test set.
  Stereoisomers:
      Found 0 Stereoisomer SMILES in the train set.
      Found 0 Stereoisomer SMILES in the test set.
```

 
## Installation


```pip install molbotomy```

This codebase uses Python 3.9 and depends on:
- [RDKit](https://www.rdkit.org/) (2023.3.2)

<!-- License-->
<h2 id="License">License</h2>

All code is under MIT license.
