from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def predict_secondary_structure(sequence):
    '''Input is a sequence; output is the predicted secondary structure content of the protein sequence.
    The fractions of different secondary structure elements present in the protein sequence.
    {Frac. of Alpha Helix, Frac. of Beta Sheet, Frac. of Turns/Loops}'''
    seq_record = SeqRecord(Seq(sequence))
    protein_analysis = ProteinAnalysis(str(seq_record.seq))
    secondary_structure = protein_analysis.secondary_structure_fraction()
    secondary_structure_dict = {'Alpha Helix': secondary_structure[0],
     'Turns Loops': secondary_structure[1],
     'Beta Sheet': secondary_structure[2]}
    return secondary_structure_dict
