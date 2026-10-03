"""Opt-in local load/text-cache smoke only; never accepts or ranks images.

Run from the repository: ./.venv/bin/python scripts/smoke_clip_runtime.py --load
"""
import argparse
import json
from importlib.resources import files

from eden_vision.classification.taxonomy import parse_taxonomy
from eden_vision.vision.runtime.clip import ClipRankingRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load', action='store_true', help='Explicitly load local weights on MPS and precompute text')
    args = parser.parse_args()
    if not args.load:
        parser.error('--load is required; no model has been loaded')
    taxonomy = parse_taxonomy(files('eden_vision.resources').joinpath('taxonomy_v2.json').read_text())
    runtime = ClipRankingRuntime(taxonomy)
    try:
        runtime.load()
        print(json.dumps(runtime.metadata, indent=2))
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
