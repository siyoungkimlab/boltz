# Pointprobe

`boltz pointprobe` probes every protein residue as a pocket for a ligand. It runs one prediction per residue, each with a pocket constraint between the ligand and that residue, and tabulates where the ligand ends up and how confident the model is.

It is faster than running `boltz predict` once per residue:

* **One MSA.** The MSA is computed once, and every prediction uses it.
* **One model load.** All the predictions run in a single `boltz predict` run, so the model is loaded once.

Each prediction still runs the whole model, since the pocket constraint is one of its inputs.

## Input

The input is a YAML file in the usual [prediction format](prediction.md). To set the pocket, give it a pocket constraint that has no `contacts`: its `binder` is the ligand to probe with, and its other fields apply to every residue.

```yaml
sequences:
  - protein:
      id: A
      sequence: ERAAMDAVCAKVDAANRLGDPLEAFPVFKKYDRNGLNVSIECKRVSGLEPATVDWAFDLTKTNMQTMYEQSEWGWKDREKREEMTDDRAWYLIAWENSSVPVAFSHFRFDVECGDEVLYCYEVQLESKVRRKGLGKFLIQILQLMANSTQMKKVMLTVFKHNHGAYQFFREALQFEIDDSSPSMSGCCGEDCSYEILSRRT
  - ligand:
      id: L
      smiles: O=C(O[C@@H]1CNCCC1)NC2=CC=C(N3C=CC=C3)C=C2
constraints:
  - pocket:
      binder: L          # the ligand to probe with
      force: true        # applies to every residue
      max_distance: 7    # applies to every residue
                         # no contacts: pointprobe fills in one residue per prediction
```

For each residue, pointprobe writes a copy of the input with `contacts: [[A, 1]]`, `[[A, 2]]`, and so on. Everything else in the input, including other constraints, is kept.

Without such a pocket constraint, pointprobe uses the default pocket (`max_distance` 6, no `force`) with the input's only ligand, or the chain given with `--binder` when there are several.

## Usage

```bash
boltz pointprobe input.yaml --use_msa_server
boltz pointprobe input.yaml --use_msa_server --probe A:30-80 --diffusion_samples 5
```

`boltz pointprobe` takes every option of `boltz predict`, plus:

| **Option** | **Type** | **Default** | **Description** |
|---|---|---|---|
| `--probe` | `TEXT` | every protein residue | The residues to probe, as comma-separated chains and residue ranges, e.g. `A,B:10-50,B:62`. Residues are numbered from 1. |
| `--binder` | `TEXT` | the input's only ligand | The chain to probe with, when the input has no pocket constraint without contacts naming it. |
| `--window` | `INTEGER` | `1` | The number of consecutive residues in each pocket: the probed residue and the ones after it in its chain. |

With `--diffusion_samples 5` and N residues, you get 5N structures.

With `--window 2`, the pocket of residue 1 is residues 1 and 2, of residue 2 is residues 2 and 3, and so on, so there is still one prediction per residue. A window never crosses into the next chain: at a chain's end it is cut short, and the last residue's pocket is that residue alone. The ligand is held within `max_distance` of every residue of its pocket. A run with `--window` above 1 gets its own output folder, `boltz_results_[input]_pointprobe_w2` for `--window 2`.

## Output

```
out_dir/boltz_results_[input]_pointprobe/
├── pointprobe_summary.csv          # One row per structure, see below
├── pointprobe_inputs/              # The input written for each residue
├── msa/                            # The MSA, computed once
├── predictions/
    ├── [input]_A_1/                # The usual prediction output for residue A1
    ├── [input]_A_2/
    ...
└── processed/
```

`pointprobe_summary.csv` has one row per structure:

| **Column** | **Description** |
|---|---|
| `probe_chain`, `probe_residue`, `probe_residue_name` | The probed residue |
| `contacts` | The residues of the pocket, e.g. `12-13` with `--window 2` |
| `model` | The model rank within that residue's predictions |
| `structure` | The structure file, in its prediction folder |
| `min_distance` | The shortest distance between any atom of the ligand and any atom of the pocket's residues, in Angstrom |
| `max_contact_distance` | The ligand's closest approach to each pocket residue, taken at the residue it came least close to, in Angstrom. This is the distance a pocket constraint limits, since it holds for every residue; with `--window 1` it equals `min_distance`. |
| `com_distance` | The distance between the centers of mass of the ligand and of the pocket's residues, in Angstrom |
| `ca_com_distance` | The distance from the pocket residues' mean CA position to the ligand's center of mass, in Angstrom. It does not depend on where the side chains point. |
| `within_max_distance` | Whether `max_contact_distance` is within the pocket's `max_distance`, that is, whether the pocket constraint holds |
| `confidence_score`, `ptm`, `iptm`, ... | The scores from the structure's confidence file |
| `affinity_pred_value`, `affinity_probability_binary` | The affinity, when the input requests it |

## Notes

* Rerunning the same command resumes: residues already predicted are skipped, and the MSA is reused.
* Probing different residues into an existing output folder also predicts any residues left unfinished from before; use a new `--out_dir` to keep runs apart.
* Boltz-1 supports one pocket constraint, at 6 Angstrom: with `--model boltz1`, the input cannot have other pocket constraints or another `max_distance`.
