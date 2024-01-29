"""
Code to clean up SMILES strings

- SpringCleaning: Class to clean up SMILES strings
- has_unfamiliar_tokens: Check if a SMILES string has unfamiliar tokens
- desalter: Get rid of salt from SMILES strings
- unrepeat smiles: If a SMILES string contains repeats of the same molecule, return a single one of them
- sanitize_mols: Sanitize a molecules with RDkit
- neutralize_mols: Use pre-defined reactions to neutralize charged molecules

Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""
from molbotomy.tools import canonicalize_smiles
from molbotomy.utils import smiles_tokenizer
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter
from tqdm.auto import tqdm


class SpringCleaning:
    """
    Class to clean up SMILES strings.

    :param canonicalize: toggles SMILES canonicalization (default = True)
    :param remove_stereochemistry: toggles stereochemistry removal (default = False)
    :param neutralize: toggles SMILES neutralization (default = True)
    :param check_for_uncommon_atoms: toggles checking for non-ochem atoms (default = True)
    :param desalt: toggles desalting (default = True)
    :param remove_solvent: toggles removal of common solvents (default = True)
    :param unrepeat_mol: toggles removal of duplicated fragments in the same SMILES (default = True)
    :param sanitize: toggles SMILES sanitization (default = True)
    """
    def __init__(self, canonicalize: bool = True, remove_stereochemistry: bool = False, neutralize: bool = True,
                 check_for_uncommon_atoms: bool = True, desalt: bool = True, remove_solvent: bool = True,
                 unrepeat_mol: bool = True, sanitize: bool = True):

        self.remove_stereo = remove_stereochemistry
        self.canonicalize = canonicalize
        self.desalt = desalt
        self.remove_solvent = remove_solvent
        self.unrepeat_mol = unrepeat_mol
        self.neutralize = neutralize
        self.sanitize = sanitize
        self.check_for_uncommon_atoms = check_for_uncommon_atoms

        self.problematic_molecules = []
        self.index_problematic_molecules = []
        self.log = []

        self.cleaned_molecules = []
        self.index_cleaned_molecules = []

    def clean(self, smiles: list[str]):
        if type(smiles) is not list:
            smiles = [smiles]

        for i, smi in tqdm(enumerate(smiles)):
            try:
                orignial_smi = smi
                # remove stereochemistry
                if self.remove_stereo:
                    smi = smi.replace('@', '')

                # desalt
                if '.' in smi and self.desalt:
                    smi = desalter(smi)

                # remove fragments
                if '.' in smi and self.remove_solvent:
                    smi = remove_common_solvents(smi)

                # remove duplicated fragments within the same SMILES
                if '.' in smi and self.unrepeat_mol:
                    smi = unrepeat_mol(smi)

                # if the SMILES is still fragmented, discard the molecule
                if '.' in smi:
                    self._fail(orignial_smi, i, 'fragmented molecule')
                    continue

                # if the SMILES contains uncommon atoms, discard the molecule
                if self.check_for_uncommon_atoms:
                    if has_unfamiliar_tokens(smi):
                        self._fail(orignial_smi, i, 'unfamiliar token')
                        continue

                # sanitize the mol
                if self.sanitize is True:
                    smi, failed_sanit = sanitize_mol(smi)
                    if failed_sanit:
                        self._fail(orignial_smi, i, 'failed sanitization')
                        continue

                # neutralize it
                if self.neutralize:
                    smi = neutralize_mol(smi)

                # finally, canonicalize
                if self.canonicalize and not self.neutralize:  # the neutralization step already canonicalizes mols
                    smi = canonicalize_smiles(smi)

                self._success(smi, i)

            except:
                self._fail(orignial_smi, i)

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
        print(f'Parsed {len(self.cleaned_molecules) + len(self.problematic_molecules)} molecules of which '
              f'{len(self.cleaned_molecules)} successfully.\nFailed to clean {len(self.problematic_molecules)} '
              f'molecules: {dict(Counter(self.log))}')


def has_unfamiliar_tokens(smiles, extra_patterns: list[str] = None) -> bool:
    """ Check if a SMILES string has unfamiliar tokens.

    :param smiles: SMILES string
    :param extra_patterns: extra tokens to consider (default = None)
        e.g. metalloids: ['Si', 'As', 'Te', 'te', 'B', 'b']  (in ChEMBL33: B+b=0.23%, Si=0.13%, As=0.01%, Te+te=0.01%).
        Mind you that the order matters. If you place 'C' before 'Cl', all Cl tokens will actually be tokenized as C,
        meaning that subsets should always come after superset strings, aka, place two letter elements first in the list.
    :return: True if the smiles string has unfamiliar tokens
    """
    tokens = smiles_tokenizer(smiles, extra_patterns)

    return len(''.join(tokens)) != len(smiles)


def desalter(smiles, salt_smarts: str = "[Cl,Na,Mg,Ca,K,Br,Zn,Ag,Al,Li,I,O,N,H]") -> str:
    """ Get rid of salt from SMILES strings, e.g., CCCCCCCCC(O)CCC(=O)[O-].[Na+] -> CCCCCCCCC(O)CCC(=O)[O-]

    :param smiles: SMILES string
    :param salt_smarts: SMARTS pattern to remove all salts (default = "[Cl,Br,Na,Zn,Mg,Ag,Al,Ca,Li,I,O,N,K,H]")
    :return: cleaned SMILES w/o salts
    """
    from rdkit.Chem.SaltRemover import SaltRemover
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

    solvents = ['O=C(O)C(F)(F)F', 'O=C(O)C(=O)O', 'O=C(O)/C=C/C(=O)O', 'CS(=O)(=O)O', 'O=C(O)/C=C\\C(=O)O', 'CC(=O)O',
                'O=S(=O)(O)O', 'O=CO', 'CCN(CC)CC', '[O-][Cl+3]([O-])([O-])[O-]', 'O=C(O)C(O)C(O)C(=O)O',
                'Cc1ccc(S(=O)(=O)[O-])cc1', 'O=C([O-])C(F)(F)F', 'Cc1ccc(S(=O)(=O)O)cc1', 'O=C(O)CC(O)(CC(=O)O)C(=O)O',
                'O=[N+]([O-])O', 'F[B-](F)(F)F', 'O=S(=O)([O-])C(F)(F)F', 'F[P-](F)(F)(F)(F)F', 'O=C(O)CCC(=O)O',
                'O=P(O)(O)O', 'NCCO', 'CS(=O)(=O)[O-]', '[O-][Cl+3]([O-])([O-])O', 'COS(=O)(=O)[O-]', 'NC(CO)(CO)CO',
                'CCO', 'CN(C)C=O', 'O=C(O)[C@H](O)[C@@H](O)C(=O)O', 'C1CCC(NC2CCCCC2)CC1', 'C', 'O=S(=O)([O-])O',
                'CNC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO', 'c1ccncc1']

    for solv in solvents:
        smiles = desalter(smiles, solv)

    return smiles


def unrepeat_mol(smiles: str):
    """ if a SMILES string contains repeats of the same molecule, return a single one of them

    :param smiles: SMILES string
    :return: unrepeated SMILES string if repeats were found, else the original SMILES string
    """
    repeats = set(smiles.split('.'))
    if len(repeats) > 1:
        return smiles
    return list(repeats)[0]


def _initialise_neutralisation_reactions() -> list[(str, str)]:
    """ adapted from the rdkit contribution of Hans de Winter """
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]


def sanitize_mol(smiles: str) -> (str, bool):
    """ Sanitize a molecules with RDkit

    :param smiles: SMILES string
    :return: SMILES string and failed_sanit flag
    """

    """ Sanitizes a molecule using rdkit """
    # init
    failed_sanit = False

    # == basic checks on SMILES validity
    mol = Chem.MolFromSmiles(smiles)

    # flags: Kekulize, check valencies, set aromaticity, conjugation and hybridization
    san_opt = Chem.SanitizeFlags.SANITIZE_ALL

    # check if the conversion to mol was successful, return otherwise
    if mol is None:
        failed_sanit = True
    # sanitization based on the flags (san_opt)
    else:
        sanitize_fail = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=san_opt)
        if sanitize_fail:
            failed_sanit = True
            raise ValueError(sanitize_fail)  # returns if failed

    return smiles, failed_sanit


def neutralize_mol(smiles: str) -> str:
    """ Use several neutralisation reactions based on patterns defined in
    _initialise_neutralisation_reactions to neutralize charged molecules

    :param smiles: SMILES string
    :return: Neutralized molecule
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


# smiles = pd.read_table("data/chembl_33_chemreps.txt").canonical_smiles.tolist()
# smiles = cleaner.problematic_molecules
# cleaner = SpringCleaning()
# cleaned_smiles = cleaner.clean(smiles[:100])

# long_smi = "CN[C@@H]1[C@H](O[C@H]2[C@H](O[C@@H]3[C@@H](O)[C@H](O)[C@@H](NC(=N)N)[C@H](O)[C@H]3NC(=N)N)O[C@@H](C)[C@]2(O)C=O)O[C@@H](CO)[C@H](O)[C@H]1O.CN[C@@H]1[C@H](O[C@H]2[C@H](O[C@@H]3[C@@H](O)[C@H](O)[C@@H](NC(=N)N)[C@H](O)[C@H]3NC(=N)N)O[C@@H](C)[C@]2(O)C=O)O[C@@H](CO)[C@H](O)[C@H]1O"
#
# # not in ChEMBL: Ge, V
# potential_salts = []
# for smi in smiles:
#     if '.' in smi:
#         potential_salts.append(smi)
#         print(smi)
#
# count*100/len(smiles)
#
# alkali_metals = ['Li', 'Na', 'K']  # done
# alkaline_earth_metals = ['Mg', 'Ca']  # done
# transition_metals = ['Ag', 'Zn']  # done
# post_transition_metals = ['Al']  # done
#
#
# # "O=C(O)[C@@H]1CCCN1.OC[C@H]1O[C@@H](c2ccc(F)c(Cc3cc4ccccc4s3)c2)[C@H](O)[C@@H](O)[C@@H]1O"
# # "Cc1c(CCN)c2cc(F)ccc2n1Cc1ccccc1.Cl"
# # "CCCCCCCCC(O)CCC(=O)[O-].[Na+]"
# # "O.O.O.O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N1CCN(CC2CC2)CC1.O=C(c1ccc(NS(=O)(=O)c2cccc3cccnc23)cc1)N1CCN(CC2CC2)CC1.O=S(=O)(O)O"
# # "O=C(C[n+]1cccc2ccccc21)C12CC3CC(CC(C3)C1)C2.[Br-]"
# # "CCC(/C=C1\Oc2ccc(-c3ccccc3)cc2C1CC)=C\c1oc2ccc(-c3ccccc3)cc2[n+]1CC.O=[N+]([O-])[O-]"
# # "CC[C@H](C)[C@H](NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](CSCC[P+](C)(C)C)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](CSCCC[P+](C)(C)C)NC(=O)[C@@H](N)CSCC[P+](C)(C)C)C(=O)N[C@@H](CSCCC[P+](C)(C)C)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(N)=O.O=C(O)C(F)(F)F.O=C([O-])C(F)(F)F.O=C([O-])C(F)(F)F.O=C([O-])C(F)(F)F.O=C([O-])C(F)(F)F"
#
# remover = SaltRemover(defnData="[Cl,Br,Na,Zn,Mg,Ag,Al,Ca,Li,I,O,N,K,H]")  # N  Ca  Zn  Mg  Ag  Al
#
# leftovers = []
# potential_solvents = []
# for smi in potential_salts:
#     # print(smi)
#     new_smi = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(smi)))
#
#     if '.' in new_smi:
#         fragments = new_smi.split('.')
#         idx = np.argmin([len(f) for f in fragments])
#         potential_solvents.append(fragments[idx])
#
#         leftovers.append(new_smi)
#         print(idx, new_smi)
#
# # I.I.OCCC1=CN(Cc2cccc(CN3C=C(CCO)NC3)n2)CN1    Cc1ccc(N2CCNCC2)c(S(=O)(=O)O)c1.O
#
# Counter(potential_solvents)
#
#
# mols_w_solv = ['NS(=O)(=O)c1nnc(NS(=O)(=O)c2ccc(-[n+]3ccccc3)cc2)s1.[O-][Cl+3]([O-])([O-])[O-]',
# 'C[n+]1cccc(NC(=O)c2ccc(C(=O)Nc3ccc[n+](C)c3)cc2)c1.Cc1ccc(S(=O)(=O)[O-])cc1.Cc1ccc(S(=O)(=O)[O-])cc1',
#                'C[n+]1c(-c2ccc(C=NNC(=O)c3ccc(C(=O)NN=Cc4ccc(-c5cn6ccccc6[n+]5C)cc4)cc3)cc2)cn2ccccc21.Cc1ccc(S(=O)(=O)[O-])cc1.Cc1ccc(S(=O)(=O)[O-])cc1',
#                ]
#
# common_solvents = ['[O-][Cl+3]([O-])([O-])[O-]', 'O=C(O)C(O)C(O)C(=O)O', 'Cc1ccc(S(=O)(=O)[O-])cc1']
#
# remover2 = SaltRemover(defnData='[O-][Cl+3]([O-])([O-])[O-] Cc1ccc(S(=O)(=O)[O-])cc1')
# mol = Chem.MolFromSmiles(mols_w_solv[0])
# res = remover2.StripMol(mol)
# print(mols_w_solv[0])
# print(Chem.MolToSmiles(res))
#
#
#
#
# for smi in leftovers:
#     for solv in common_solvents:
#         if solv in smi:
#             print(smi)
#
# # solvents
# # [O-][Cl+3]([O-])([O-])[O-]
# # O=C(O)C(O)C(O)C(=O)O
# # Cc1ccc(S(=O)(=O)[O-])cc1
# # O=C([O-])C(F)(F)F
# # Cc1ccc(S(=O)(=O)O)cc1
# # O=C(O)CC(O)(CC(=O)O)C(=O)O
# # O=[N+]([O-])O
# # F[B-](F)(F)F
# # O=S(=O)([O-])C(F)(F)F
# # F[P-](F)(F)(F)(F)F
# # O=C(O)CCC(=O)O
# # O=P(O)(O)O
# # NCCO
# # CS(=O)(=O)[O-]
# # [O-][Cl+3]([O-])([O-])O
# # COS(=O)(=O)[O-]
# # NC(CO)(CO)CO
# # CCO
# # CN(C)C=O
# # O=C(O)[C@H](O)[C@@H](O)C(=O)O
# # C1CCC(NC2CCCCC2)CC1
# # C
# # O=S(=O)([O-])O
# # CNC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO
# # c1ccncc1
#



