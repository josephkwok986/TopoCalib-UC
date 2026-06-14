"""Ablation variant switches."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VariantConfig:
    name: str
    use_train_calibration: bool
    use_inference_calibration: bool
    use_local_evidence: bool
    use_ambiguity_gate: bool
    use_part_prior: bool

    def as_dict(self) -> dict[str, bool | str]:
        return asdict(self)


def variant_config(name: str) -> VariantConfig:
    key = name.upper()
    table = {
        "B0": VariantConfig(key, False, False, False, False, False),
        "B1": VariantConfig(key, False, True, True, False, False),
        "B2": VariantConfig(key, True, False, True, False, False),
        "B3": VariantConfig(key, True, True, True, False, False),
        "B4": VariantConfig(key, True, True, True, True, False),
        "B5": VariantConfig(key, True, True, True, True, True),
    }
    if key not in table:
        raise ValueError(f"Unknown variant {name!r}; expected one of {sorted(table)}")
    return table[key]

