import json
import pickle
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from rdkit.Chem import Mol
from torch import Tensor

from boltz.data.types import Coords, Interface, Record, Structure, StructureV2
from boltz.data.write.dms import to_dms
from boltz.data.write.mae import to_mae
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
from boltz.data.write.properties import confidence_properties
from boltz.timing import PredictionTimer, read_preprocessing_timing, write_timing

# Formats that write bond orders and formal charges, read from the reference
# molecules rather than from the structure alone.
CHEMISTRY_FORMATS = ("mae", "dms")


class TimedPredictionWriter(BasePredictionWriter):
    """A prediction writer that times the stages of each prediction."""

    def __init__(self) -> None:
        super().__init__(write_interval="batch")
        self.timer = PredictionTimer()

    def on_predict_start(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,
    ) -> None:
        """Time the stages of the model about to predict."""
        self.timer.attach(pl_module)
        self.timer.start_run()

    def on_predict_batch_start(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,
        batch: dict[str, Tensor],  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Start timing a prediction."""
        self.timer.start_prediction(pl_module.device)


class BoltzWriter(TimedPredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: Literal["pdb", "mmcif", "mae", "dms"] = "mmcif",
        boltz2: bool = False,
        write_embeddings: bool = False,
        *,
        mol_dir: Optional[str] = None,
        extra_mols_dir: Optional[str] = None,
        ccd_path: Optional[str] = None,
        timing_dir: Optional[str] = None,
        inputs_dir: Optional[str] = None,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.
        mol_dir : str, optional
            The CCD molecules directory, read for the MAE and DMS formats.
        extra_mols_dir : str, optional
            The processed molecules of each record (its SMILES ligands), read
            for the MAE and DMS formats.
        ccd_path : str, optional
            The CCD dictionary, read for the MAE and DMS formats when a residue
            is not in ``mol_dir``. Boltz-1 downloads only this.
        timing_dir : str, optional
            The preprocessing times of each record, reported in its timing file.
        inputs_dir : str, optional
            The copy of each record's input file, saved with its predictions.

        """
        super().__init__()
        if output_format not in ["pdb", "mmcif", *CHEMISTRY_FORMATS]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0
        self.boltz2 = boltz2
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_embeddings = write_embeddings
        self.mol_dir = None if mol_dir is None else Path(mol_dir)
        self.extra_mols_dir = None if extra_mols_dir is None else Path(extra_mols_dir)
        self.ccd_path = None if ccd_path is None else Path(ccd_path)
        self.ccd: Optional[dict[str, Mol]] = None
        self.ccd_mols: dict[str, Optional[Mol]] = {}
        self.timing_dir = None if timing_dir is None else Path(timing_dir)
        self.inputs_dir = None if inputs_dir is None else Path(inputs_dir)

    def load_molecules(self, record_id: str, structure: Structure) -> dict[str, Mol]:
        """Get the reference molecule of every residue in a structure.

        These are the molecules the featurizer read the atoms from: the record's
        own (its SMILES ligands) first, then the CCD. A residue found in neither
        is left out, and written from what the structure stores.
        """
        molecules = {}
        if self.extra_mols_dir is not None:
            path = self.extra_mols_dir / f"{record_id}.pkl"
            if path.exists():
                with path.open("rb") as f:
                    molecules.update(pickle.load(f))  # noqa: S301

        for name in set(structure.residues["name"].tolist()) - set(molecules):
            if name not in self.ccd_mols:
                self.ccd_mols[name] = self.load_ccd_molecule(name)
            if self.ccd_mols[name] is not None:
                molecules[name] = self.ccd_mols[name]
        return molecules

    def load_ccd_molecule(self, name: str) -> Optional[Mol]:
        """Get a CCD molecule from ``mol_dir``, or else from the CCD dictionary."""
        if self.mol_dir is not None:
            path = self.mol_dir / f"{name}.pkl"
            if path.exists():
                with path.open("rb") as f:
                    return pickle.load(f)  # noqa: S301

        # Boltz-1 has no mol_dir, so its CCD dictionary is loaded, once, instead.
        if self.ccd is None and self.ccd_path is not None and self.ccd_path.exists():
            with self.ccd_path.open("rb") as f:
                self.ccd = pickle.load(f)  # noqa: S301
        return None if self.ccd is None else self.ccd.get(name)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        write_start = time.perf_counter()
        if prediction["exception"]:
            self.failed += 1
            self.timer.end_prediction()
            return

        # Get the records
        records: list[Record] = batch["record"]

        # Get the predictions
        coords = prediction["coords"]
        coords = coords.unsqueeze(0)

        pad_masks = prediction["masks"]

        # Get ranking
        if "confidence_score" in prediction:
            argsort = torch.argsort(prediction["confidence_score"], descending=True)
            idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}
        # Handles cases where confidence summary is False
        else:
            idx_to_rank = {i: i for i in range(len(records))}

        # Iterate over the records
        for record, coord, pad_mask in zip(records, coords, pad_masks):
            # Load the structure
            path = self.data_dir / f"{record.id}.npz"
            if self.boltz2:
                structure: StructureV2 = StructureV2.load(path)
            else:
                structure: Structure = Structure.load(path)

            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains completely
            structure = structure.remove_invalid_chains()

            # Load the reference molecules, for the formats that need them
            molecules = {}
            if self.output_format in CHEMISTRY_FORMATS:
                molecules = self.load_molecules(record.id, structure)

            for model_idx in range(coord.shape[0]):
                # Get model coord
                model_coord = coord[model_idx]
                # Unpad
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()

                # New atom table
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True
                if self.boltz2:
                    structure: StructureV2
                    coord_unpad = [(x,) for x in coord_unpad]
                    coord_unpad = np.array(coord_unpad, dtype=Coords)

                # Mew residue table
                residues = structure.residues
                residues["is_present"] = True

                # Update the structure
                interfaces = np.array([], dtype=Interface)
                if self.boltz2:
                    new_structure: StructureV2 = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                        coords=coord_unpad,
                    )
                else:
                    new_structure: Structure = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                    )

                # Update chain info
                chain_info = []
                for chain in new_structure.chains:
                    old_chain_idx = chain_map[chain["asym_id"]]
                    old_chain_info = record.chains[old_chain_idx]
                    new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                    chain_info.append(new_chain_info)

                # Save the structure
                struct_dir = self.output_dir / record.id
                struct_dir.mkdir(exist_ok=True)

                # Get plddt's
                plddts = None
                if "plddt" in prediction:
                    plddts = prediction["plddt"][model_idx]

                # Create path name
                outname = f"{record.id}_model_{idx_to_rank[model_idx]}"

                # Get the confidence summary, also written into MAE and DMS
                confidence_summary_dict = None
                properties = None
                if "plddt" in prediction:
                    confidence_summary_dict = {}
                    for key in [
                        "confidence_score",
                        "ptm",
                        "iptm",
                        "ligand_iptm",
                        "protein_iptm",
                        "complex_plddt",
                        "complex_iplddt",
                        "complex_pde",
                        "complex_ipde",
                    ]:
                        confidence_summary_dict[key] = prediction[key][model_idx].item()
                    confidence_summary_dict["chains_ptm"] = {
                        idx: prediction["pair_chains_iptm"][idx][idx][model_idx].item()
                        for idx in prediction["pair_chains_iptm"]
                    }
                    confidence_summary_dict["pair_chains_iptm"] = {
                        idx1: {
                            idx2: prediction["pair_chains_iptm"][idx1][idx2][
                                model_idx
                            ].item()
                            for idx2 in prediction["pair_chains_iptm"][idx1]
                        }
                        for idx1 in prediction["pair_chains_iptm"]
                    }
                    properties = confidence_properties(
                        confidence_summary_dict, new_structure
                    )

                # Save the structure
                if self.output_format == "pdb":
                    path = struct_dir / f"{outname}.pdb"
                    with path.open("w") as f:
                        f.write(
                            to_pdb(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                elif self.output_format == "mmcif":
                    path = struct_dir / f"{outname}.cif"
                    with path.open("w") as f:
                        f.write(
                            to_mmcif(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                elif self.output_format == "mae":
                    path = struct_dir / f"{outname}.mae"
                    with path.open("w") as f:
                        f.write(
                            to_mae(
                                new_structure,
                                molecules,
                                plddts=plddts,
                                boltz2=self.boltz2,
                                title=outname,
                                properties=properties,
                            )
                        )
                elif self.output_format == "dms":
                    path = struct_dir / f"{outname}.dms"
                    to_dms(
                        path,
                        new_structure,
                        molecules,
                        plddts=plddts,
                        boltz2=self.boltz2,
                        title=outname,
                        properties=properties,
                    )
                else:
                    path = struct_dir / f"{outname}.npz"
                    np.savez_compressed(path, **asdict(new_structure))

                if self.boltz2 and record.affinity and idx_to_rank[model_idx] == 0:
                    path = struct_dir / f"pre_affinity_{record.id}.npz"
                    np.savez_compressed(path, **asdict(new_structure))
                    np.array(atoms["coords"][:, None], dtype=Coords)

                # Save confidence summary
                if "plddt" in prediction:
                    path = (
                        struct_dir
                        / f"confidence_{record.id}_model_{idx_to_rank[model_idx]}.json"
                    )
                    with path.open("w") as f:
                        f.write(
                            json.dumps(
                                confidence_summary_dict,
                                indent=4,
                            )
                        )

                    # Save plddt
                    plddt = prediction["plddt"][model_idx]
                    path = (
                        struct_dir
                        / f"plddt_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, plddt=plddt.cpu().numpy())

                # Save pae
                if "pae" in prediction:
                    pae = prediction["pae"][model_idx]
                    path = (
                        struct_dir
                        / f"pae_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pae=pae.cpu().numpy())

                # Save pde
                if "pde" in prediction:
                    pde = prediction["pde"][model_idx]
                    path = (
                        struct_dir
                        / f"pde_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pde=pde.cpu().numpy())
                
            # Save embeddings
            if self.write_embeddings and "s" in prediction and "z" in prediction:
                s = prediction["s"].cpu().numpy()
                z = prediction["z"].cpu().numpy()

                path = (
                    struct_dir
                    / f"embeddings_{record.id}.npz"
                )
                np.savez_compressed(path, s=s, z=z)

        # Save a copy of the input, for the record
        if self.inputs_dir is not None and self.inputs_dir.exists():
            for record in records:
                for source in self.inputs_dir.iterdir():
                    if source.stem == record.id:
                        shutil.copy2(source, self.output_dir / record.id / source.name)

        # Save timing
        self.timer.add("write_outputs", time.perf_counter() - write_start)
        for record in records:
            path = self.output_dir / record.id / f"timing_{record.id}.json"
            preprocessing = read_preprocessing_timing(self.timing_dir, record.id)
            write_timing(path, self.timer.report(record.id, preprocessing))
        self.timer.end_prediction()

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201


class BoltzAffinityWriter(TimedPredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__()
        self.failed = 0
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        write_start = time.perf_counter()
        if prediction["exception"]:
            self.failed += 1
            self.timer.end_prediction()
            return
        # Dump affinity summary
        affinity_summary = {}
        pred_affinity_value = prediction["affinity_pred_value"]
        pred_affinity_probability = prediction["affinity_probability_binary"]
        affinity_summary = {
            "affinity_pred_value": pred_affinity_value.item(),
            "affinity_probability_binary": pred_affinity_probability.item(),
        }
        if "affinity_pred_value1" in prediction:
            pred_affinity_value1 = prediction["affinity_pred_value1"]
            pred_affinity_probability1 = prediction["affinity_probability_binary1"]
            pred_affinity_value2 = prediction["affinity_pred_value2"]
            pred_affinity_probability2 = prediction["affinity_probability_binary2"]
            affinity_summary["affinity_pred_value1"] = pred_affinity_value1.item()
            affinity_summary["affinity_probability_binary1"] = (
                pred_affinity_probability1.item()
            )
            affinity_summary["affinity_pred_value2"] = pred_affinity_value2.item()
            affinity_summary["affinity_probability_binary2"] = (
                pred_affinity_probability2.item()
            )

        # Save the affinity summary
        struct_dir = self.output_dir / batch["record"][0].id
        struct_dir.mkdir(exist_ok=True)
        path = struct_dir / f"affinity_{batch['record'][0].id}.json"

        with path.open("w") as f:
            f.write(json.dumps(affinity_summary, indent=4))

        # Save timing
        self.timer.add("write_outputs", time.perf_counter() - write_start)
        record_id = batch["record"][0].id
        path = struct_dir / f"timing_affinity_{record_id}.json"
        write_timing(path, self.timer.report(record_id, preprocessing=None))
        self.timer.end_prediction()

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
