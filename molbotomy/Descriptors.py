"""
Compute some molecular descriptors from RDkit molecules

- rdkit_to_array: helper function to convert RDkit fingerprints to a numpy array
- mols_to_maccs: Get MACCs key descriptors from a list of RDKit molecule objects
- mols_to_ecfp: Get ECFPs from a list of RDKit molecule objects
- mols_to_descriptors: Get the full set of available RDKit descriptors (normalized) for a list of RDKit molecule objects


Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Union


def rdkit_to_array(fp: list) -> np.ndarray:
    """ Convert a list of RDkit fingerprint objects into a numpy array """
    from rdkit.DataStructs import ConvertToNumpyArray

    output = []
    for f in fp:
        arr = np.zeros((1,))
        ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)


def mols_to_maccs(mols: list, progressbar: bool = False, to_array: bool = False) -> Union[list, np.ndarray]:
    """ Get MACCs key descriptors from a list of RDKit molecule objects

    :param mols: RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param progressbar: toggles progressbar (default = False)
    :param to_array: Toggles conversion of RDKit fingerprint objects to a Numpy Array (default = False)
    :return: Numpy Array of MACCs keys
    """
    from rdkit.Chem import MACCSkeys

    fp = [MACCSkeys.GenMACCSKeys(m) for m in tqdm(mols, disable=not progressbar)]
    if not to_array:
        return fp
    return rdkit_to_array(fp)


def mols_to_ecfp(mols: list, radius: int = 2, nbits: int = 1024, progressbar: bool = False, to_array: bool = False) \
        -> Union[list, np.ndarray]:
    """ Get ECFPs from a list of RDKit molecule objects

    :param mols: RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param radius: Radius of the ECFP (default = 2)
    :param nbits: Number of bits (default = 1024)
    :param progressbar: toggles progressbar (default = False)
    :param to_array: Toggles conversion of RDKit fingerprint objects to a Numpy Array (default = False)
    :return: list of RDKit ECFP fingerprint objects, or a Numpy Array of ECFPs if to_array=True
    """
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

    fp = [GetMorganFingerprintAsBitVect(m, radius, nBits=nbits) for m in tqdm(mols, disable=not progressbar)]
    if not to_array:
        return fp
    return rdkit_to_array(fp)


def mols_to_descriptors(mols: list, progressbar: bool = False, normalize: bool = True) -> np.ndarray:
    """ Get the full set of available RDKit descriptors for a list of RDKit molecule objects

    :param mols: RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param progressbar: toggles progressbar (default = False)
    :param normalize: toggles min-max normalization
    :return: Numpy Array of all RDKit descriptors
    """
    from rdkit.Chem import Descriptors
    from sklearn.preprocessing import normalize as normi

    x = np.array(pd.DataFrame([Descriptors.CalcMolDescriptors(m) for m in tqdm(mols, disable=not progressbar)]))
    if normalize:
        x = normi(x, axis=0, norm='max')

    return x
