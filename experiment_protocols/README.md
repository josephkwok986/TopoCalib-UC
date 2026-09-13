# Experiment protocol manifests

Generated split, labeled-part, subset, and workload manifests may be stored under this directory when a release includes the fixed protocol files. The command-line tools also accept manifests kept with external result artifacts.

All manifests use dataset-relative part IDs. Face records use `(dataset, part_id, face_id)` with a zero-based, part-local face index. Loaders validate dataset identity, class mapping, duplicate IDs, split membership, and the relevant face-count or budget constraints before use.

`sdbscan_table_b6.json` records the Table B6 validation grid, fixed approximation parameters, dataset-specific selected values, and execution seed rule.
