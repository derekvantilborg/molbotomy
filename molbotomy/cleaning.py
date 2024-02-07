"""
Code to clean up SMILES strings

- SpringCleaning: Class to clean up SMILES strings
- clean_mol: Cleans up a molecule
- has_unfamiliar_tokens: Check if a SMILES string has unfamiliar tokens
- flatten_stereochemistry: Get rid of stereochemistry in a SMILES string
- desalter: Get rid of salt from SMILES strings
- remove_common_solvents: Get rid of some of the most commonly used solvents in a SMILES string
- unrepeat_smiles: If a SMILES string contains repeats of the same molecule, return a single one of them
- sanitize_mols: Sanitize a molecules with RDkit
- neutralize_mols: Use pre-defined reactions to neutralize charged molecules

Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""
from molbotomy.tools import canonicalize_smiles
from molbotomy.utils import smiles_tokenizer
from molbotomy.constants import SOLVENTS, NEUTRALIZATION_PATTERNS
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter
from typing import Union
from tqdm.auto import tqdm
import warnings


class SpringCleaning:
    """
    Class to clean up SMILES strings.

    :param canonicalize: toggles SMILES canonicalization (default = True)
    :param flatten_stereochem: toggles stereochemistry removal (default = False)
    :param neutralize: toggles SMILES neutralization (default = True)
    :param check_for_uncommon_atoms: toggles checking for non-ochem atoms (default = True)
    :param desalt: toggles desalting (default = True)
    :param remove_solvent: toggles removal of common solvents (default = True)
    :param unrepeat: toggles removal of duplicated fragments in the same SMILES (default = True)
    :param sanitize: toggles SMILES sanitization (default = True)
    """
    def __init__(self, canonicalize: bool = True, flatten_stereochem: bool = False, neutralize: bool = True,
                 check_for_uncommon_atoms: bool = True, desalt: bool = True, remove_solvent: bool = True,
                 unrepeat: bool = True, sanitize: bool = True):

        self.settings = {'canonicalize': canonicalize, 'flatten_stereochem': flatten_stereochem, 'desalt': desalt,
                         'neutralize': neutralize, 'check_for_uncommon_atoms': check_for_uncommon_atoms,
                         'remove_solvent': remove_solvent, 'unrepeat': unrepeat, 'sanitize': sanitize}

        self.problematic_molecules = []
        self.index_problematic_molecules = []
        self.log = []

        self.cleaned_molecules = []
        self.index_cleaned_molecules = []

    def clean(self, smiles: list[str], progressbar: bool = True) -> list[str]:
        """ Clean a list of SMILES strings

        :param smiles: SMILES that need cleaning
        :param progressbar: toggles the progressbar
        :return: a list of all successfully parsed SMILES strings
        """
        if type(smiles) is not list:
            smiles = [smiles]

        for i, smi in tqdm(enumerate(smiles), disable=not progressbar):
            try:
                cleaned_smi, log = clean_mol(smi, **self.settings)
                if cleaned_smi is None:
                    self._fail(smi, i, log)
                else:
                    self._success(cleaned_smi, i)
            except:
                self._fail(smi, i, 'unknown')

        self.summary()
        return self.cleaned_molecules

    def _fail(self, smiles, index, reason: str = 'unknown') -> None:
        self.problematic_molecules.append(smiles)
        self.index_problematic_molecules.append(index)
        self.log.append(reason)

    def _success(self, smiles, index) -> None:
        self.cleaned_molecules.append(smiles)
        self.index_cleaned_molecules.append(index)

    def summary(self) -> None:
        if len(self.problematic_molecules) > 0:
            print(f'Parsed {len(self.cleaned_molecules) + len(self.problematic_molecules)} molecules of which '
                  f'{len(self.cleaned_molecules)} successfully.\nFailed to clean {len(self.problematic_molecules)} '
                  f'molecules: {dict(Counter(self.log))}')
        else:
            print(f'Parsed {len(self.cleaned_molecules)} molecules successfully.')


def clean_mol(smiles: str, canonicalize: bool = True, flatten_stereochem: bool = False, neutralize: bool = True,
              check_for_uncommon_atoms: bool = True, desalt: bool = True, remove_solvent: bool = True,
              unrepeat: bool = True, sanitize: bool = True) -> (str, str):
    """ Clean a SMILES string by canonicalizing, neutralizing, and getting rid of junk

    :param smiles: SMILES string
    :param canonicalize: toggles SMILES canonicalization (default = True)
    :param flatten_stereochem: toggles stereochemistry removal (default = False)
    :param neutralize: toggles SMILES neutralization (default = True)
    :param check_for_uncommon_atoms: toggles checking for non-ochem atoms (default = True)
    :param desalt: toggles desalting (default = True)
    :param remove_solvent: toggles removal of common solvents (default = True)
    :param unrepeat: toggles removal of duplicated fragments in the same SMILES (default = True)
    :param sanitize: toggles SMILES sanitization (default = True)

    :return: cleaned SMILES string, log message
    """

    # get rid of stereochemistry
    if flatten_stereochem:
        smiles = flatten_stereochemistry(smiles)

    # canonicalize
    if canonicalize:
        smiles = canonicalize_smiles(smiles)
        if smiles is None:
            return None, 'failed canonicalization'

    # desalt
    if desalt:
        smiles = desalter(smiles)

    # remove fragments
    if remove_solvent:
        smiles = remove_common_solvents(smiles)

    # remove duplicated fragments within the same SMILES
    if unrepeat:
        smiles = unrepeat_smiles(smiles)

    # if the SMILES is still fragmented, discard the molecule
    if '.' in smiles:
        return None, 'fragmented SMILES'

    # if the SMILES contains uncommon atoms, discard the molecule
    if check_for_uncommon_atoms:
        if has_unfamiliar_tokens(smiles):
            return None, 'unfamiliar token'

    # sanitize the mol
    if sanitize:
        smiles = sanitize_mol(smiles)
        if smiles is None:
            return None, 'failed sanitization'

    # neutralize it
    if neutralize:
        smiles = neutralize_mol(smiles)

    return smiles, 'successful'


def has_unfamiliar_tokens(smiles, extra_patterns: list[str] = None) -> bool:
    """ Check if a SMILES string has unfamiliar tokens.

    :param smiles: SMILES string
    :param extra_patterns: extra tokens to consider (default = None)
        e.g. metalloids: ['Si', 'As', 'Te', 'te', 'B', 'b']  (in ChEMBL33: B+b=0.23%, Si=0.13%, As=0.01%, Te+te=0.01%).
        Mind you that the order matters. If you place 'C' before 'Cl', all Cl tokens will actually be tokenized as C,
        meaning that subsets should always come after superset strings, aka, place two letter elements first in the list
    :return: True if the smiles string has unfamiliar tokens
    """
    tokens = smiles_tokenizer(smiles, extra_patterns)

    return len(''.join(tokens)) != len(smiles)


def flatten_stereochemistry(smiles: str) -> str:
    """ Remove stereochemistry from a SMILES string """
    return smiles.replace('@', '')


def desalter(smiles, salt_smarts: str = "[Cl,Na,Mg,Ca,K,Br,Zn,Ag,Al,Li,I,O,N,H]") -> str:
    """ Get rid of salt from SMILES strings, e.g., CCCCCCCCC(O)CCC(=O)[O-].[Na+] -> CCCCCCCCC(O)CCC(=O)[O-]

    :param smiles: SMILES string
    :param salt_smarts: SMARTS pattern to remove all salts (default = "[Cl,Br,Na,Zn,Mg,Ag,Al,Ca,Li,I,O,N,K,H]")
    :return: cleaned SMILES w/o salts
    """
    if '.' not in smiles:
        return smiles

    remover = SaltRemover(defnData=salt_smarts)

    new_smi = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(smiles)))

    return new_smi


def remove_common_solvents(smiles: str) -> str:
    """ Remove commonly used solvents from a SMILES string, e.g.,
    Nc1ncnc2scc(-c3ccc(NC(=O)Cc4cc(F)ccc4F)cc3)c12.O=C(O)C(F)(F)F -> Nc1ncnc2scc(-c3ccc(NC(=O)Cc4cc(F)ccc4F)cc3)c12

     The following solvents are removed:

    'O=C(O)C(F)(F)F', 'O=C(O)C(=O)O', 'O=C(O)/C=C/C(=O)O', 'CS(=O)(=O)O', 'O=C(O)/C=C\\C(=O)O', 'CC(=O)O',
    'O=S(=O)(O)O', 'O=CO', 'CCN(CC)CC', '[O-][Cl+3]([O-])([O-])[O-]', 'O=C(O)C(O)C(O)C(=O)O',
    'Cc1ccc(S(=O)(=O)[O-])cc1', 'O=C([O-])C(F)(F)F', 'Cc1ccc(S(=O)(=O)O)cc1', 'O=C(O)CC(O)(CC(=O)O)C(=O)O',
    'O=[N+]([O-])O', 'F[B-](F)(F)F', 'O=S(=O)([O-])C(F)(F)F', 'F[P-](F)(F)(F)(F)F', 'O=C(O)CCC(=O)O', 'O=P(O)(O)O',
    'NCCO', 'CS(=O)(=O)[O-]', '[O-][Cl+3]([O-])([O-])O', 'COS(=O)(=O)[O-]', 'NC(CO)(CO)CO', 'CCO', 'CN(C)C=O',
    'O=C(O)[C@H](O)[C@@H](O)C(=O)O', 'C1CCC(NC2CCCCC2)CC1', 'C', 'O=S(=O)([O-])O',
    'CNC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO', 'c1ccncc1'

     (not the most efficient code out there)
    :param smiles: SMILES string
    :return: cleaned SMILES
    """
    if '.' not in smiles:
        return smiles

    for solv in SOLVENTS:
        smiles = desalter(smiles, solv)

    return smiles


def unrepeat_smiles(smiles: str) -> str:
    """ if a SMILES string contains repeats of the same molecule, return a single one of them

    :param smiles: SMILES string
    :return: unrepeated SMILES string if repeats were found, else the original SMILES string
    """
    if '.' not in smiles:
        return smiles

    repeats = set(smiles.split('.'))
    if len(repeats) > 1:
        return smiles
    return list(repeats)[0]


def _initialise_neutralisation_reactions() -> list[(str, str)]:
    """ adapted from the rdkit contribution of Hans de Winter """
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in NEUTRALIZATION_PATTERNS]


def sanitize_mol(smiles: str) -> Union[str, None]:
    """ Sanitize a molecules with RDkit

    :param smiles: SMILES string
    :return: SMILES string if sanitized or None if failed sanitizing
    """
    # basic checks on SMILES validity
    mol = Chem.MolFromSmiles(smiles)

    # flags: Kekulize, check valencies, set aromaticity, conjugation and hybridization
    san_opt = Chem.SanitizeFlags.SANITIZE_ALL

    if mol is not None:
        sanitize_error = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=san_opt)
        if sanitize_error:
            warnings.warn(sanitize_error)
            return None
    else:
        return None

    return Chem.MolToSmiles(mol)


def neutralize_mol(smiles: str) -> str:
    """ Use several neutralisation reactions based on patterns defined in NEUTRALIZATION_PATTERNS to neutralize charged
    molecules

    :param smiles: SMILES string
    :return: SMILES of the neutralized molecule
    """
    mol = Chem.MolFromSmiles(smiles)

    # retrieves the transformations
    transfm = _initialise_neutralisation_reactions()  # set of transformations

    # applies the transformations
    for i, (reactant, product) in enumerate(transfm):
        while mol.HasSubstructMatch(reactant):
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]

    # converts back the molecule to smiles
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    return smiles
