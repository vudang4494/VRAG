"""Unit tests for canonical benchmark script dataset loading and formatting."""

import json

from scripts.benchmark import load_dataset


def test_load_dataset_list(tmp_path):
    """Verify loading dataset formatted as JSON list."""
    dataset_file = tmp_path / "test_list.json"
    data = [{"question": "Q1", "reference": "A1"}, {"question": "Q2", "reference": "A2"}]
    dataset_file.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_dataset(str(dataset_file))
    assert len(loaded) == 2
    assert loaded[0]["question"] == "Q1"


def test_load_dataset_dict(tmp_path):
    """Verify loading dataset formatted as JSON object with 'samples' or 'prompts' key."""
    dataset_file = tmp_path / "test_dict.json"
    data = {"samples": [{"user_input": "UI1", "reference_answer": "RA1"}]}
    dataset_file.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_dataset(str(dataset_file))
    assert len(loaded) == 1
    assert loaded[0]["user_input"] == "UI1"
