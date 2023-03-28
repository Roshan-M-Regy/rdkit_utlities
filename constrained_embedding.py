########################################################################
# Author: Roshan M Regy
# Email ID: roshanm.regy@gmail.com
# Take a ligand structure from a pdb file and constrain newer smiles
# strings based on Maximum Common substructure 
# Parts of code taken from Iwatobipen's and RDKit blogs. 
########################################################################

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    prog="Constrained Embedding using RDKit",
    description="Generates and constrains new ligands based on MCS with reference ligand in PDB",
)
parser.add_argument(
    '-r',
    '--refpdb',
    required=True,
    type=str,
    help='name of reference ligand PDB file'
)
parser.add_argument(
    '-s',
    '--refsmiles',
    required=True,
    type=str,
    help="Smiles string of reference ligand for bond order assignment"
)
parser.add_argument(
    '-q',
    '--querysmiles',
    required=True,
    type=str,
    help='name of file with smiles strings of query ligand saved in column Smiles'
)
parser.add_argument(
    '-n',
    '--numconfs',
    required=False,
    type=int,
    default=1,
    help='Number of conformers to generate per query ligand'
)
parser.add_argument(
    '-o',
    '--outputprefix',
    type=str,
    required=False,
    default='query_mol',
    help="prefix of output mol files"
)
args = parser.parse_args()

def make_reference(pdbfilename, smiles):
    reference_frompdb = Chem.MolFromPDBFile(pdbfilename)
    reference_frompdb = Chem.RemoveHs(reference_frompdb)
    template = (Chem.RemoveHs(Chem.MolFromSmiles(smiles)))
    reference = AllChem.AssignBondOrdersFromTemplate(template, reference_frompdb)
    return reference

def constrain_embed(query_smiles, reference,numconfs=1,outputname='query_mol'):
    query_mol = Chem.AddHs(Chem.MolFromSmiles(query_smiles))
    mcs = rdFMCS.FindMCS([reference, query_mol])
    mcsmol = Chem.MolFromSmarts(mcs.smartsString)
    rwmol = Chem.RWMol(mcsmol)
    rwconf = Chem.Conformer(rwmol.GetNumAtoms())
    matches = rwmol.GetSubstructMatch(mcsmol)
    refconf = reference.GetConformer()
    refmatch = reference.GetSubstructMatch(mcsmol)
    for i,match in enumerate(matches):
        rwconf.SetAtomPosition(match,refconf.GetAtomPosition(refmatch[i]))
    rwmol.AddConformer(rwconf)
    
    cid = 0
    for seed in [int(x) for x in np.random.randint(1,100001, numconfs)]:
        AllChem.EmbedMolecule(query_mol,randomSeed=seed)
        AllChem.UFFOptimizeMolecule(query_mol)
        try:
            AllChem.ConstrainedEmbed(query_mol,rwmol,randomseed=seed)
            Chem.MolToMolFile(query_mol,'%s%s_conf%s.mol'%(outputname,i+1,cid))
            cid+=1
        except:
            continue
    return query_mol


smileset = pd.read_csv(args.querysmiles)
numconfs = args.numconfs
reference = make_reference(args.refpdb,args.refsmiles)
for i,smiles in enumerate(smileset.Smiles):
    query_mol = constrain_embed(smiles,reference,args.numconfs,args.outputprefix)