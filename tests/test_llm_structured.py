import pytest
from pydantic import BaseModel

from agentops.errors import MalformedOutputError
from agentops.llm import Message
from agentops.llm.structured import generate_structured
from agentops.testing.fakes import FakeLLM

MESSAGES = [Message(role="user", content="give me a toy as JSON")]


class Toy(BaseModel):
    name: str
    count: int


def test_valid_json_on_first_try():
    llm = FakeLLM(['{"name": "robot", "count": 3}'])

    toy, response = generate_structured(llm, MESSAGES, Toy)

    assert toy == Toy(name="robot", count=3)
    assert response.text == '{"name": "robot", "count": 3}'
    assert len(llm.calls) == 1  # no repair needed


def test_repairs_once_after_invalid_json():
    llm = FakeLLM(
        [
            "not json at all",  # first attempt: fails to parse
            '{"name": "robot", "count": 3}',  # repair attempt: valid
        ]
    )

    toy, _ = generate_structured(llm, MESSAGES, Toy)

    assert toy == Toy(name="robot", count=3)
    assert len(llm.calls) == 2
    # the repair call includes the original messages plus one feedback message
    assert len(llm.calls[1]) == len(llm.calls[0]) + 1


def test_repairs_once_after_schema_mismatch():
    llm = FakeLLM(
        [
            '{"name": "robot"}',  # missing required field "count"
            '{"name": "robot", "count": 3}',  # repair attempt: valid
        ]
    )

    toy, _ = generate_structured(llm, MESSAGES, Toy)

    assert toy == Toy(name="robot", count=3)


def test_gives_up_after_repair_also_fails():
    llm = FakeLLM(["still not json", "still not json either"])

    with pytest.raises(MalformedOutputError):
        generate_structured(llm, MESSAGES, Toy)

    assert len(llm.calls) == 2  # exactly one repair attempt, then stop


def test_original_messages_are_not_mutated():
    original = list(MESSAGES)
    llm = FakeLLM(['{"name": "robot", "count": 3}'])

    generate_structured(llm, original, Toy)

    assert original == MESSAGES  # generate_structured must not mutate the caller's list
