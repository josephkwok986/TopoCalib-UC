"""Part-level split and low-label sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random

from .partgraph_dataset import PartGraphCache


@dataclass(frozen=True)
class PartSplit:
    train: list[str]
    val: list[str]
    test: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def class_aware_subset(
    cache: PartGraphCache,
    *,
    min_parts_per_class: int,
    seed: int,
) -> list[str]:
    """Greedily select a small part subset where every class appears in N parts."""

    if min_parts_per_class <= 0:
        raise ValueError("min_parts_per_class must be positive.")
    rng = random.Random(seed)
    target = {cls: min_parts_per_class for cls in cache.classes}
    counts = {cls: 0 for cls in cache.classes}
    remaining = list(cache.part_ids)
    rng.shuffle(remaining)
    selected: list[str] = []

    while any(counts[cls] < target[cls] for cls in cache.classes):
        best_idx = -1
        best_score: tuple[int, int, float] | None = None
        for idx, part_id in enumerate(remaining):
            labels = cache.by_part_id(part_id).class_set
            gain = sum(1 for cls in labels if counts.get(cls, 0) < target.get(cls, 0))
            rare_gain = sum(max(target.get(cls, 0) - counts.get(cls, 0), 0) for cls in labels)
            if gain == 0:
                continue
            score = (gain, rare_gain, rng.random())
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            missing = [cls for cls in cache.classes if counts[cls] < target[cls]]
            raise ValueError(f"Cannot satisfy min_parts_per_class={min_parts_per_class}; missing classes: {missing}")
        part_id = remaining.pop(best_idx)
        selected.append(part_id)
        for cls in cache.by_part_id(part_id).class_set:
            if cls in counts:
                counts[cls] += 1
    return selected


def random_part_split(
    cache: PartGraphCache,
    *,
    part_ids: list[str] | None = None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    require_train_all_classes: bool = True,
    max_tries: int = 1000,
) -> PartSplit:
    """Random part-level split, optionally retrying until train covers all classes."""

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    ids = list(part_ids) if part_ids is not None else list(cache.part_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("part_ids contains duplicates.")
    all_classes = cache.labels_for_parts(set(ids))
    if set(cache.classes) - all_classes:
        missing = sorted(set(cache.classes) - all_classes)
        raise ValueError(f"Selected parts do not cover global classes: {missing}")

    for offset in range(max_tries):
        rng = random.Random(seed + offset)
        shuffled = ids[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1 if n >= 3 else 0
        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]
        if not test and n >= 2:
            test = [train.pop()]
        if require_train_all_classes and cache.labels_for_parts(set(train)) != all_classes:
            continue
        return PartSplit(train=sorted(train), val=sorted(val), test=sorted(test))
    raise ValueError("Failed to create a train split covering all classes; increase data or adjust ratios.")


def dataset_split(cache: PartGraphCache) -> PartSplit:
    train = sorted(record.part_id for record in cache.records if record.split == "train")
    val = sorted(record.part_id for record in cache.records if record.split in {"val", "valid", "validation"})
    test = sorted(record.part_id for record in cache.records if record.split == "test")
    assigned = set(train) | set(val) | set(test)
    if len(assigned) != len(cache.records):
        missing = sorted(set(cache.part_ids) - assigned)
        raise ValueError(f"Dataset split is incomplete or unsupported for parts: {missing[:10]}")
    return PartSplit(train=train, val=val, test=test)


def sample_labeled_parts(
    cache: PartGraphCache,
    *,
    train_part_ids: list[str],
    budget: int,
    seed: int,
    required_classes: list[int] | None = None,
) -> list[str]:
    """Select B labeled training parts with mandatory class coverage."""

    if budget <= 0:
        raise ValueError("budget must be positive.")
    ids = list(train_part_ids)
    if budget > len(ids):
        raise ValueError(f"budget={budget} exceeds train parts={len(ids)}")
    required = set(required_classes if required_classes is not None else cache.classes)
    available = cache.labels_for_parts(set(ids))
    if required - available:
        raise ValueError(f"Train split cannot cover classes: {sorted(required - available)}")

    rng = random.Random(seed)
    remaining = ids[:]
    rng.shuffle(remaining)
    selected: list[str] = []
    covered: set[int] = set()
    while not required.issubset(covered):
        best_idx = -1
        best_score: tuple[int, float] | None = None
        for idx, part_id in enumerate(remaining):
            labels = cache.by_part_id(part_id).class_set
            gain = len((labels & required) - covered)
            if gain == 0:
                continue
            score = (gain, rng.random())
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            raise ValueError(f"Cannot cover required classes: {sorted(required - covered)}")
        selected_part = remaining.pop(best_idx)
        selected.append(selected_part)
        covered |= cache.by_part_id(selected_part).class_set
        if len(selected) > budget:
            raise ValueError("Internal sampling error: selected more parts than budget.")
        if len(selected) == budget and not required.issubset(covered):
            raise ValueError(
                f"budget={budget} is too small to cover classes; missing {sorted(required - covered)}"
            )

    while len(selected) < budget:
        selected.append(remaining.pop())
    return sorted(selected)


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative.")
    total = train_ratio + val_ratio + test_ratio
    if not 0.999 <= total <= 1.001:
        raise ValueError(f"split ratios must sum to 1, got {total}")

