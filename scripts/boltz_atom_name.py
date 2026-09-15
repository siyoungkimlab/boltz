"""Show the names Boltz gives a SMILES ligand's atoms, for a covalent bond constraint.

usage: python boltz_atom_name.py "<SMILES>" [--png out.png]
e.g.   python boltz_atom_name.py "CCC(=O)Nc1ccccc1" --png ligand.png
       python boltz_atom_name.py "[CH3:1]CC(=O)Nc1ccccc1" --png ligand.png

Prints every heavy atom's name; with --png, draws the ligand with the names on
it. To single out the bonding atom, optionally mark it with an atom map
number, :1, written inside brackets with its hydrogens, as in [CH3:1]: it is
then reported on its own and highlighted in the drawing.
"""

import argparse
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def boltz_names(smiles: str) -> Chem.Mol:
    """Name the atoms exactly as Boltz does for a SMILES ligand."""
    mol = AllChem.AddHs(AllChem.MolFromSmiles(smiles))
    for atom, rank in zip(mol.GetAtoms(), AllChem.CanonicalRankAtoms(mol)):
        atom.SetProp("name", atom.GetSymbol().upper() + str(rank + 1))
    return mol


def draw(mol: Chem.Mol, highlight: list[int], path: str) -> None:
    """Draw the heavy atoms, each labelled with its Boltz name."""
    heavy = Chem.RemoveHs(mol)
    AllChem.Compute2DCoords(heavy)
    for atom in heavy.GetAtoms():
        atom.SetProp("atomNote", atom.GetProp("name"))
    drawer = rdMolDraw2D.MolDraw2DCairo(900, 650)
    options = drawer.drawOptions()
    options.annotationFontScale = 0.9
    options.addStereoAnnotation = False
    options.setHighlightColour((1.0, 0.55, 0.2))
    drawer.DrawMolecule(
        heavy,
        highlightAtoms=highlight,
        highlightAtomRadii=dict.fromkeys(highlight, 0.5),
    )
    drawer.FinishDrawing()
    with Path(path).open("wb") as f:
        f.write(drawer.GetDrawingText())


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "smiles",
    help="the ligand SMILES, optionally with the bonding atom marked as [..:1]",
)
parser.add_argument("--png", help="draw the ligand with its atom names to this PNG")
args = parser.parse_args()

marked = Chem.MolFromSmiles(args.smiles)
if marked is None:
    sys.exit("RDKit cannot read that SMILES.")
tagged = [a.GetIdx() for a in marked.GetAtoms() if a.GetAtomMapNum() == 1]
if len(tagged) > 1:
    sys.exit("Mark at most one atom as the bonding atom, e.g. [CH3:1].")
for atom in marked.GetAtoms():
    atom.SetAtomMapNum(0)
# A map number changes RDKit's canonical ranking, so name the atoms on the
# unmarked SMILES: that is the SMILES to put in the YAML.
plain = Chem.MolToSmiles(marked, canonical=False)
mol = boltz_names(plain)
heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
print(f"smiles for the YAML: {plain}")  # noqa: T201
print("heavy atoms:        " + "  ".join(a.GetProp("name") for a in heavy))  # noqa: T201
if tagged:
    print(f"bonding atom name:  {mol.GetAtomWithIdx(tagged[0]).GetProp('name')}")  # noqa: T201
if args.png:
    draw(mol, tagged, args.png)
    print(f"drawing:            {args.png}")  # noqa: T201
