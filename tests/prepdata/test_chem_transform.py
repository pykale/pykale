from kale.prepdata.chem_transform import integer_label_protein, integer_label_smiles


def test_chem_wrong_smiles():
    wrong_smiles = "NS(=O)(=O)*c1cc2C"
    assert len(integer_label_smiles(wrong_smiles))


def test_chem_illegal_character():
    smiles = "O=C(c1ccc(F)cc1)C1CCN(CCC2Cc3cc(F)ccc3C2=O)CC1*"
    target = "MEILCEDNISLSSIPNSLM*QLGDGPRL"
    drug_encoding = integer_label_smiles(smiles)
    target_encoding = integer_label_protein(target)
    assert drug_encoding.size == 85
    assert target_encoding.size == 1200
