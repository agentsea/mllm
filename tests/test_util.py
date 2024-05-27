import json
from typing import Any, Dict

import pytest

from mllm.util import extract_parse_json


def test_valid_json_in_backticks():
    input_str = 'Here is some data:\n```json\n{"key": "value", "number": 123}\n```'
    expected_output = {"key": "value", "number": 123}
    assert extract_parse_json(input_str) == expected_output


def test_valid_nested_json_in_backticks():
    input_str = 'Nested JSON example:\n```json\n{"outer": {"inner": "value"}, "array": [1, 2, 3]}\n```'
    expected_output = {"outer": {"inner": "value"}, "array": [1, 2, 3]}
    assert extract_parse_json(input_str) == expected_output


def test_invalid_json_in_backticks():
    input_str = 'Invalid JSON:\n```json\n{"key": "value", "number": }\n```'
    with pytest.raises(json.JSONDecodeError):
        extract_parse_json(input_str)


def test_plain_json_without_backticks():
    input_str = '{"key": "value", "number": 123}'
    expected_output = {"key": "value", "number": 123}
    assert extract_parse_json(input_str) == expected_output


def test_plain_invalid_json():
    input_str = '{"key": "value", "number": }'
    with pytest.raises(json.JSONDecodeError):
        extract_parse_json(input_str)


def test_json_with_additional_text():
    input_str = 'Here is some data:\n```json\n{"key": "value", "number": 123}\n```\nAnd some more text.'
    expected_output = {"key": "value", "number": 123}
    assert extract_parse_json(input_str) == expected_output


def test_no_json_in_input():
    input_str = "There is no JSON here."
    with pytest.raises(json.JSONDecodeError):
        extract_parse_json(input_str)
