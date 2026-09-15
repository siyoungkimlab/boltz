# Screen

`boltz screen` predicts a complex with each ligand of a list: the same protein, one ligand at a time. It tabulates the confidence of each prediction and, when requested, its affinity.

It is faster than running `boltz predict` once per ligand:

* **One MSA.** The MSA is computed once, and every prediction uses it.
* **One model load.** All the predictions run in a single `boltz predict` run, so the model is loaded once, and so is the affinity model.

## Input

The input is a YAML template in the usual [prediction format](prediction.md), with one ligand entry. For each ligand of the list, screen replaces that entry's SMILES and keeps everything else, such as a pocket constraint on the ligand or an affinity property:

```yaml
sequences:
  - protein:
      id: A
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
  - ligand:
      id: L
      smiles: C              # replaced by each ligand of the list
constraints:                 # optional: holds every ligand in the same pocket
  - pocket:
      binder: L
      contacts: [[A, 42], [A, 44], [A, 68]]
properties:                  # optional: predicts every ligand's affinity (Boltz-2)
  - affinity:
      binder: L
```

When the template has several ligand entries, choose the one to replace with `--ligand`.

The ligand list is either a `.smi` or `.txt` file, with a SMILES and optionally a name on each line:

```
Nc1ccc(cc1)C(=O)O     aminobenzoate
CC(=O)Oc1ccccc1C(=O)O aspirin
c1ccc2[nH]ccc2c1
```

or a `.csv` file with a `smiles` column and optionally a `name` column. A ligand without a name is called `lig_001`, `lig_002`, and so on. Names become folder names, so characters other than letters, digits, `_` and `-` are replaced with `_`, and repeated names get `_2`, `_3`. Every SMILES is checked with RDKit before anything runs: ligands it cannot read are listed and skipped.

## Usage

```bash
boltz screen template.yaml --ligands ligands.smi --use_msa_server
boltz screen template.yaml --ligands ligands.csv --use_msa_server --diffusion_samples 5 --devices 4
```

`boltz screen` takes every option of `boltz predict`, plus:

| **Option** | **Type** | **Default** | **Description** |
|---|---|---|---|
| `--ligands` | `PATH` | required | The ligand list: `.smi`, `.txt` or `.csv`. |
| `--ligand` | `TEXT` | the template's only ligand | The template's ligand chain to replace. |

## Output

```
out_dir/boltz_results_[template]_screen/
├── screen_summary.csv              # One row per structure, see below
├── screen_inputs/                  # The input written for each ligand
├── msa/                            # The MSA, computed once
├── predictions/
    ├── aminobenzoate/              # The usual prediction output for each ligand
    ├── aspirin/
    ...
└── processed/
```

`screen_summary.csv` has one row per structure:

| **Column** | **Description** |
|---|---|
| `ligand` | The ligand's name |
| `model` | The model rank within that ligand's predictions |
| `structure` | The structure file, in its prediction folder |
| `input_smiles` | The ligand's SMILES, as given |
| `predicted_smiles` | The ligand's SMILES as predicted, with its stereochemistry read from the predicted coordinates. From MAE and DMS output it is the ligand exactly as the file holds it: its atoms, formal charges and bond orders. PDB and mmCIF keep no bond orders or charges, so there the input's chemistry is placed on the predicted coordinates. |
| `matches_input` | Whether the predicted ligand is the input: the same atoms, bonds and charges, and every stereocenter and double bond geometry the input SMILES specifies (unspecified ones are ignored). For the affinity binder, compared with the SMILES as Boltz standardizes it, which neutralizes charges. |
| `min_distance`, `max_contact_distance`, `com_distance`, `ca_com_distance`, `within_max_distance` | When the template has a pocket constraint on the ligand: how close the ligand came to the pocket's residues, as in [pointprobe](pointprobe.md) |
| `confidence_score`, `ptm`, `iptm`, `ligand_iptm`, ... | The scores from the structure's confidence file |
| `affinity_pred_value`, `affinity_probability_binary` | The affinity, when the template requests it |

## Notes

* Rerunning the same command resumes: ligands already predicted are skipped, and the MSA is reused.
* Boltz reuses an input it has processed before, by name. Screen therefore refuses a ligand whose SMILES changed since the last run into the same `--out_dir`: rename it or use a new `--out_dir`.
* Affinity is only predicted by Boltz-2. With `--model boltz1`, the template cannot request it, and may have at most one pocket constraint, at 6 Angstrom.
* A bond constraint to the screened ligand is refused: each ligand names its atoms differently, so one constraint cannot name the same atom in all of them. Other bonds, such as a disulfide, are kept, and their atoms are checked before running.
