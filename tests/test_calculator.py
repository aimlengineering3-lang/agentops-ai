import pytest

from agentops.tools import CalculatorError, safe_eval


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", 4.0),
        ("133396 + 5300 + 44631", 183327.0),
        ("(10 + 5) * 3", 45.0),
        ("10 / 4", 2.5),
        ("2 ** 10", 1024.0),
        ("7 // 2", 3.0),
        ("-5 + 3", -2.0),
        ("10 % 3", 1.0),
    ],
)
def test_evaluates_basic_arithmetic(expression, expected):
    assert safe_eval(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "[x for x in range(10)]",
        "os.system('ls')",
        "1 if True else 2",
        "'a' + 'b'",
        "len([1, 2, 3])",
        "x + 1",
        "",
        "2 +",
        "2 ** 2 ** 2 ** 2 ** 2",  # legal syntax, absurd magnitude via nested power
    ],
)
def test_rejects_anything_that_is_not_plain_arithmetic(expression):
    with pytest.raises(CalculatorError):
        safe_eval(expression)


def test_rejects_division_by_zero():
    with pytest.raises(CalculatorError):
        safe_eval("1 / 0")


def test_rejects_expression_over_the_length_limit():
    with pytest.raises(CalculatorError):
        safe_eval("1+" * 150 + "1")


def test_rejects_huge_single_power():
    with pytest.raises(CalculatorError):
        safe_eval("10 ** 100")
