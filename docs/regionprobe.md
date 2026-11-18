# Regionprobe

`boltz regionprobe` probes every region of a protein as a pocket for a ligand. Where [pointprobe](pointprobe.md) runs one prediction per residue, regionprobe runs one per region: it predicts the protein without the ligand first, splits that structure into k regions, and runs one prediction per region, each with a pocket constraint between the ligand and the residues at the heart of that region.

It runs in two stages:

1. **Apo.** The input is predicted without its binder, with the same options. Every protein residue is placed by its CB atom (CA for glycine), and those atoms, from all the protein chains together, are split into k regions by k-means. A region near an interface holds residues of several chains.
2. **Probing.** For each region, the residues nearest its center, up to `--pocket_size` of them, become the `contacts` of a pocket constraint, and the complex is predicted with it. All k predictions share one MSA and one loaded model, as in pointprobe.

The k-means uses a fixed random state, so the same apo structure always gives the same regions, and a rerun resumes.

## Input

The input is a YAML file in the usual [prediction format](prediction.md), set up as for pointprobe: a pocket constraint without `contacts` names the ligand to probe with as its `binder`, and its other fields apply to every region.

```yaml
sequences:
  - protein:
      id: A
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
  - ligand:
      id: L
      smiles: Nc1ccc(cc1)C(=O)O
constraints:
  - pocket:
      binder: L          # the ligand to probe with
      max_distance: 7    # applies to every region
                         # no contacts: regionprobe fills in each region's residues
```

Without such a pocket constraint, the default pocket (`max_distance` 6, no `force`) is used with the input's only ligand, or the chain given with `--binder`.

The apo input is the same file without the binder: its chain is removed, along with every constraint and property that refers to it (the pocket, a bond to it, an affinity on it). Other chains, such as a cofactor, are kept.

## Usage

```bash
boltz regionprobe input.yaml --use_msa_server
boltz regionprobe input.yaml --use_msa_server --regions 40 --pocket_size all --diffusion_samples 5
```

`boltz regionprobe` takes every option of `boltz predict`, plus:

| **Option** | **Type** | **Default** | **Description** |
|---|---|---|---|
| `--regions` | `INTEGER` | `25` | The number of regions, k: one prediction each. |
| `--pocket_size` | `INTEGER` or `all` | `10` | The residues in each pocket: the ones of the region nearest its center, up to this many, or `all` for every residue of the region. |
| `--binder` | `TEXT` | the input's only ligand | The chain to probe with, when the input has no pocket constraint without contacts naming it. |
| `--apo` | `PATH` | none | A structure to read the regions from instead of predicting one: a Boltz output with the input's chain ids and residue numbers, such as an earlier run's `boltz_results_apo/predictions/[input]_apo/[input]_apo_model_0.mae`. Runs with other `--pocket_size` values then share the same regions, and skip the apo prediction. The map takes this file's format. |

With `--diffusion_samples 5` and k regions, you get 5k structures, plus the apo prediction's. A run with another `--regions` or `--pocket_size` than the defaults gets its own output folder, such as `boltz_results_[input]_regionprobe_k40_mall`, since it makes other pockets under the same names.

## Output

```
out_dir/boltz_results_[input]_regionprobe/
├── regionprobe_summary.csv         # One row per structure, see below
├── regions.csv                     # Every residue's region and rank, see below
├── regions.mae                     # The apo structure with the regions in its B-factors
├── regions.pml                     # Colors them in PyMOL: pymol regions.pml
├── regionprobe_inputs/             # The input written for each region, and apo/
├── boltz_results_apo/              # The apo prediction, a usual boltz predict output
├── msa/                            # The MSA, computed once
├── predictions/
    ├── [input]_region01/           # The usual prediction output for region 1
    ├── [input]_region02/
    ...
└── processed/
```

### The region map

The map is the apo structure (its top-ranked model) in the run's own `--output_format`, so `regions.mae` by default, `regions.dms`, `regions.cif` or `regions.pdb` otherwise, with every protein residue's region number in the B-factor, 0 for the other chains, and an occupancy of 1 for the residues that became a pocket and 0.5 for the rest. Nothing else is changed: a MAE or DMS map keeps its bonds, charges and the confidence properties. Any viewer can color it by B-factor; `regions.pml` does it in PyMOL, with a numbered sphere at each region's center and the pocket residues as sticks:

```bash
pymol regions.pml
```

PyMOL does not read DMS, so no script is written for a DMS run; VMD reads the map.

`regions.csv` has one row per residue:

| **Column** | **Description** |
|---|---|
| `chain`, `residue`, `residue_name` | The residue |
| `region` | Its region, 1 to k. Regions are numbered in sequence order of their nearest residue. |
| `rank` | Its rank within the region by distance to the center, 1 for the nearest |
| `center_distance` | The distance from its CB (CA for glycine) to the region's center, in Å |
| `pocket` | Whether it is one of the region's pocket residues |

### The summary

`regionprobe_summary.csv` has one row per structure, with the same SMILES, distance, bond and score columns as the [pointprobe summary](pointprobe.md#output), where the pocket's residues are the region's pocket residues, and:

| **Column** | **Description** |
|---|---|
| `region` | The region probed |
| `region_residues` | The number of residues in the region |
| `pocket_residues` | The pocket, e.g. `A:12;A:15;B:40` |
| `region_center_distance` | The distance from the ligand's center of mass to the region's center, in Å |

## Notes

* Rerunning the same command resumes: the apo prediction and the regions already predicted are skipped, and the MSA is reused. The regions are computed again from the same apo structure, so they come out the same.
* The apo stage is a `boltz predict` run of its own, so the model is loaded twice in all.
* Boltz-1 supports one pocket constraint, at 6 Angstrom: with `--model boltz1`, the input cannot have other pocket constraints or another `max_distance`.
* A bond constraint, such as a covalent ligand's, is kept in every probing input and dropped from the apo one. Its atoms are checked before running.
