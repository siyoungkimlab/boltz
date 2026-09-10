"""Properties of a whole structure, for the MAE and DMS writers.

Both formats store properties of the structure alongside its atoms: an MAE
file in its ``f_m_ct`` block, a DMS file in its ``msys_ct`` table. The names
here are what msys reads from either one, so ``boltz_confidence_score`` is
``r_boltz_confidence_score`` in an MAE file and the ``boltz_confidence_score``
column of a DMS file.
"""

import re

from boltz.data.types import Structure


def _chain_label(chain_names: dict[int, str], index: object) -> str:
    """Name a chain by its name in the structure, fit for a property name."""
    name = chain_names.get(int(index), str(index))
    return re.sub(r"\W", "_", name)


def confidence_properties(confidence: dict, structure: Structure) -> dict[str, float]:
    """Flatten a confidence summary into named structure properties.

    The scores keep their names from the confidence file, prefixed with
    ``boltz_``. The per-chain scores, keyed there by chain index, are named by
    chain instead: ``boltz_chain_ptm_A`` and ``boltz_pair_chains_iptm_A_B``.
    """
    chain_names = {
        int(chain["asym_id"]): str(chain["name"]) for chain in structure.chains
    }
    properties = {}
    for key, value in confidence.items():
        if key == "chains_ptm":
            for index, score in value.items():
                label = _chain_label(chain_names, index)
                properties[f"boltz_chain_ptm_{label}"] = float(score)
        elif key == "pair_chains_iptm":
            for index_1, row in value.items():
                for index_2, score in row.items():
                    label_1 = _chain_label(chain_names, index_1)
                    label_2 = _chain_label(chain_names, index_2)
                    properties[f"boltz_pair_chains_iptm_{label_1}_{label_2}"] = float(
                        score
                    )
        else:
            properties[f"boltz_{key}"] = float(value)
    return properties
