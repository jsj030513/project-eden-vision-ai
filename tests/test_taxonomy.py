import json
from importlib.resources import files

import pytest

from eden_vision.classification.taxonomy import load_taxonomy, parse_taxonomy


def seed():
    return json.loads(files("eden_vision.resources").joinpath("taxonomy_v1.json").read_text())


def test_seed_taxonomy_structure():
    taxonomy = load_taxonomy()
    concepts = {c.concept_id: c for c in taxonomy.concepts}
    assert taxonomy.taxonomy_version == "eden-v1"
    assert len(concepts) == 25
    assert concepts["rose"].parent_id == "flower"
    assert concepts["flower"].parent_id == "plant"
    assert concepts["plant"].parent_id == "living_thing"
    assert concepts["rose"].broad_category_id == "plant"
    assert concepts["artifact"].display_name == "Human-made object"
    assert concepts["artifact"].parent_id is None
    assert "UNKNOWN" not in concepts


@pytest.mark.parametrize("case", ["duplicate", "missing_parent", "cycle", "broad", "missing_broad",
                                 "shape", "id", "aliases", "descriptions", "version", "empty"])
def test_reject_invalid_taxonomy(case):
    data = seed()
    first = data["concepts"][0]
    if case == "duplicate":
        data["concepts"].append(first.copy())
    elif case == "missing_parent":
        first["parent_id"] = "missing"
    elif case == "cycle":
        first["parent_id"] = "rose"
    elif case == "broad":
        first["broad_category_id"] = "furniture"
    elif case == "missing_broad":
        first["broad_category_id"] = "missing"
    elif case == "shape":
        del first["display_name"]
    elif case == "id":
        first["concept_id"] = "Invalid ID"
    elif case in {"aliases", "descriptions"}:
        first[case] = "not a list"
    elif case == "version":
        data["taxonomy_version"] = ""
    else:
        data["concepts"] = []
    with pytest.raises(ValueError):
        parse_taxonomy(json.dumps(data))


def test_invalid_json():
    with pytest.raises(ValueError):
        parse_taxonomy("{broken")


def test_alias_and_description_extension():
    data = seed()
    data["concepts"][0].update(aliases=["organism"], descriptions=["A living organism."])
    taxonomy = parse_taxonomy(json.dumps(data))
    assert taxonomy.concepts[0].aliases == ("organism",)
    assert taxonomy.concepts[0].descriptions == ("A living organism.",)
