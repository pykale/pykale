"""
Functions for labeling and encoding chemical characters like Compound SMILES and atom string, refer to
https://github.com/hkmztrk/DeepDTA and https://github.com/thinng/GraphDTA.
"""

import logging

import numpy as np
from rdkit import Chem

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25

CHARISOSMISET = {
    "#": 29,
    "%": 30,
    ")": 31,
    "(": 1,
    "+": 32,
    "-": 33,
    "/": 34,
    ".": 2,
    "1": 35,
    "0": 3,
    "3": 36,
    "2": 4,
    "5": 37,
    "4": 5,
    "7": 38,
    "6": 6,
    "9": 39,
    "8": 7,
    "=": 40,
    "A": 41,
    "@": 8,
    "C": 42,
    "B": 9,
    "E": 43,
    "D": 10,
    "G": 44,
    "F": 11,
    "I": 45,
    "H": 12,
    "K": 46,
    "M": 47,
    "L": 13,
    "O": 48,
    "N": 14,
    "P": 15,
    "S": 49,
    "R": 16,
    "U": 50,
    "T": 17,
    "W": 51,
    "V": 18,
    "Y": 52,
    "[": 53,
    "Z": 19,
    "]": 54,
    "\\": 20,
    "a": 55,
    "c": 56,
    "b": 21,
    "e": 57,
    "d": 22,
    "g": 58,
    "f": 23,
    "i": 59,
    "h": 24,
    "m": 60,
    "l": 25,
    "o": 61,
    "n": 26,
    "s": 62,
    "r": 27,
    "u": 63,
    "t": 28,
    "y": 64,
}

CHARISOSMILEN = 64

CHARATOMSET = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "Unknown",
]

CHARATOMLEN = 44


def integer_label_smiles(smiles, max_length=85, isomeric=False):
    """
    Integer encoding for SMILES string sequence.

    Args:
        smiles (str): Simplified molecular-input line-entry system, which is a specification in the form of a line
        notation for describing the structure of chemical species using short ASCII strings.
        max_length (int): Maximum encoding length of input SMILES string. (default: 85)
        isomeric (bool): Whether the input SMILES string includes isomeric information (default: False).
    """
    if not isomeric:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"rdkit cannot find this SMILES {smiles}.")
            return None
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(smiles[:max_length]):
        try:
            encoding[idx] = CHARISOSMISET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in default SMILE category encoding, skip and treat as " f"padding."
            )

    return encoding


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.

    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string. (default: 1200)
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding
