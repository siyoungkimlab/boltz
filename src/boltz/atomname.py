"""boltz atomname: show the names Boltz gives a SMILES ligand's atoms.

A covalent bond constraint names each of its atoms. Boltz names a SMILES
ligand's atoms itself: element and canonical rank, with hydrogens added, so
the names cannot be read off the SMILES. This command prints them and draws
the ligand labelled with them; the same naming checks the bond constraints of
``boltz pointprobe`` and ``boltz screen`` before they run.
"""

from pathlib import Path

import click
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def boltz_atom_names(smiles: str) -> Chem.Mol:
    """Return the ligand with its atoms named exactly as Boltz names them.

    Boltz adds hydrogens and ranks the atoms canonically; each atom is named by
    its element and its rank plus one, such as ``C21``.
    """
    mol = AllChem.AddHs(AllChem.MolFromSmiles(smiles))
    for atom, rank in zip(mol.GetAtoms(), AllChem.CanonicalRankAtoms(mol)):
        atom.SetProp("name", atom.GetSymbol().upper() + str(rank + 1))
    return mol


def named_smiles(smiles: str, affinity: bool) -> str:
    """Return the SMILES Boltz names a ligand's atoms from.

    Boltz standardizes the affinity binder's SMILES first, which can change its
    charges and so its names.
    """
    if not affinity:
        return smiles
    from boltz.data.parse.schema import standardize  # noqa: PLC0415

    return standardize(smiles)


def heavy_atom_names(smiles: str, affinity: bool = False) -> list[str]:
    """Return the names Boltz gives a SMILES ligand's heavy atoms."""
    mol = boltz_atom_names(named_smiles(smiles, affinity))
    return [atom.GetProp("name") for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


def draw_atom_names(mol: Chem.Mol, highlight: list[int], path: Path) -> None:
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
    with path.open("wb") as f:
        f.write(drawer.GetDrawingText())


@click.command(
    "atomname", short_help="Show the names Boltz gives a SMILES ligand's atoms."
)
@click.argument("smiles")
@click.option(
    "--png",
    type=click.Path(dir_okay=False),
    default=None,
    help="Draw the ligand, labelled with its atom names, to this PNG file.",
)
@click.option(
    "--affinity",
    is_flag=True,
    help=(
        "Name the atoms as Boltz does for the affinity binder, whose SMILES it "
        "standardizes first."
    ),
)
def atomname(smiles: str, png: str, affinity: bool) -> None:
    """Show the names Boltz gives a SMILES ligand's atoms.

    For a covalent bond constraint, which names one of the ligand's atoms, as
    in [L, 1, C21]. To single out the bonding atom, mark it with an atom map
    number, :1, written inside brackets with its hydrogens, as in
    [CH3:1]CC(=O)Nc1ccccc1: it is then reported and highlighted.
    """
    marked = Chem.MolFromSmiles(smiles)
    if marked is None:
        msg = "RDKit cannot read that SMILES."
        raise click.BadParameter(msg, param_hint="SMILES")
    tagged = [a.GetIdx() for a in marked.GetAtoms() if a.GetAtomMapNum() == 1]
    if len(tagged) > 1:
        msg = "Mark at most one atom as the bonding atom, e.g. [CH3:1]."
        raise click.BadParameter(msg, param_hint="SMILES")
    if tagged and affinity:
        msg = (
            "--affinity standardizes the SMILES, which can reorder its atoms: "
            "leave the mark out and find the atom in the drawing."
        )
        raise click.UsageError(msg)

    # A map number changes RDKit's canonical ranking, so name the atoms on the
    # unmarked SMILES: that is the SMILES to put in the YAML.
    for atom in marked.GetAtoms():
        atom.SetAtomMapNum(0)
    plain = Chem.MolToSmiles(marked, canonical=False)
    try:
        source = named_smiles(plain, affinity)
    except ValueError as error:
        msg = f"Boltz cannot standardize that SMILES for affinity: {error}"
        raise click.UsageError(msg) from error
    mol = boltz_atom_names(source)

    click.echo(f"smiles for the YAML: {plain}")
    if source != plain:
        click.echo(f"named from:          {source} (as Boltz standardizes it)")
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    click.echo("heavy atoms:         " + "  ".join(a.GetProp("name") for a in heavy))
    if tagged:
        name = mol.GetAtomWithIdx(tagged[0]).GetProp("name")
        click.echo(f"bonding atom name:   {name}")
    if png:
        draw_atom_names(mol, tagged, Path(png))
        click.echo(f"drawing:             {png}")
