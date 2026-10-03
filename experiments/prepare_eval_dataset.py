"""Register user-provided evaluation photos without interpreting their content."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.eval_dataset import PromotionError, dataset_status, import_inbox, init_dataset, promote_draft, set_dataset_taxonomy, update_dataset_count
from experiments.model_runtime import load_evaluation_taxonomy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/Users/dev/Datasets/eden-vision-eval"))
    parser.add_argument("--taxonomy", choices=("eden-v1", "eden-v2"), default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create a new empty dataset; refuse any existing root")
    init.add_argument("--dataset-id", required=True)
    sub.add_parser("import", help="validate and copy inbox files into images/")
    sub.add_parser("promote", help="promote completed human labels from draft to canonical manifest")
    sub.add_parser("status", help="show inbox/imported/draft/canonical counts")
    sub.add_parser("update-count", help="synchronize dataset.json image_count with manifest.jsonl")
    sub.add_parser("set-taxonomy", help="set taxonomy before any annotations exist")
    args = parser.parse_args()
    selected = load_evaluation_taxonomy(args.taxonomy) if args.taxonomy else None
    if args.command == "init":
        if selected is None:
            parser.error("init requires explicit --taxonomy eden-v1|eden-v2")
        print(json.dumps(init_dataset(args.dataset, args.dataset_id, selected), indent=2))
    elif args.command == "import":
        print(json.dumps(import_inbox(args.dataset, taxonomy=selected), indent=2))
    elif args.command == "promote":
        if selected is None:
            raise SystemExit("promote requires --taxonomy eden-v1|eden-v2")
        try:
            result = promote_draft(args.dataset, selected)
        except PromotionError as exc:
            print(json.dumps(exc.summary, indent=2))
            return 1
        print(json.dumps(result, indent=2))
    elif args.command == "status":
        print(json.dumps(dataset_status(args.dataset), indent=2))
    elif args.command == "set-taxonomy":
        if selected is None:
            raise SystemExit("set-taxonomy requires --taxonomy eden-v1|eden-v2")
        print(json.dumps({"taxonomy_version": set_dataset_taxonomy(args.dataset, selected)}, indent=2))
    else:
        print(json.dumps({"image_count": update_dataset_count(args.dataset)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
