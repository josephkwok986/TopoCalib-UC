#!/usr/bin/env python3
"""Import UV-Net raw outputs as unified result JSON."""

from __future__ import annotations

import argparse

from baselines.external.common.result_import import add_import_args, import_external_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_import_args(parser, method="uvnet", dataset="fusion360_gallery")
    import_external_result(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
