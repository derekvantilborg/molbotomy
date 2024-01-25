"""
A collection of utility functions

- canonicalize_smiles: Canonicalize a list of SMILES strings with the RDKit SMILES canonicalization algorithm
- smiles_to_mols: Convert a list of SMILES strings to RDkit molecules (and sanitize them)
- mols_to_smiles: Convert a list of RDkit molecules back into SMILES strings
- mols_to_scaffolds: Convert a list of RDKit molecules objects into scaffolds (bismurcko or bismurcko_generic)
- map_scaffolds: Find which molecules share the same scaffold


Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict


def canonicalize_smiles(smiles: list[str]) -> list[str]:
    """ Canonicalize a list of SMILES strings with the RDKit SMILES canonicalization algorithm """
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]


def smiles_to_mols(smiles: list[str], sanitize: bool = True) -> list:
    """ Convert a list of SMILES strings to RDkit molecules (and sanitize them)

    :param smiles: List of SMILES strings
    :param sanitize: toggles sanitization of the molecule. Defaults to True.
    :return: List of RDKit mol objects
    """
    mols = []
    for smi in smiles:
        molecule = Chem.MolFromSmiles(smi, sanitize=sanitize)

        # If sanitization is unsuccessful, catch the error, and try again without
        # the sanitization step that caused the error
        if sanitize:
            flag = Chem.SanitizeMol(molecule, catchErrors=True)
            if flag != Chem.SanitizeFlags.SANITIZE_NONE:
                Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        Chem.rdPartialCharges.ComputeGasteigerCharges(molecule)

        mols.append(molecule)

    return mols


def mols_to_smiles(mols) -> list[str]:
    return [Chem.MolToSmiles(m) for m in mols]


def mols_to_scaffolds(mols: list, scaffold_type: str = 'bismurcko') -> list:
    """ Convert a list of RDKit molecules objects into scaffolds (bismurcko or bismurcko_generic)

    :param mols: RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param scaffold_type: type of scaffold: bismurcko, bismurcko_generic (default = 'bismurcko')
    :return: RDKit mol objects of the scaffolds
    """
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric

    if scaffold_type == 'bismurcko_generic':
        scaffolds = [MakeScaffoldGeneric(m) for m in mols]
    else:
        scaffolds = [GetScaffoldForMol(m) for m in mols]

    return scaffolds


def map_scaffolds(mols: list) -> (list, dict[str, list[int]]):
    """ Find which molecules share the same scaffold

    :param mols: RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :return: scaffolds, dict of unique scaffolds and which molecules (indices) share them -> {'c1ccccc1': [0, 12, 47]}
    """

    scaffolds = mols_to_scaffolds(mols)

    uniques = defaultdict(list)
    for i, s in enumerate(scaffolds):
        uniques[Chem.MolToSmiles(s)].append(i)

    return scaffolds, uniques




