import pytest

from src import calculator


@pytest.mark.parametrize(
    "x, y, expected",
    [(2, 3, 5), (5, 0, 5), (-1, 1, 0), (-1, -1, -2), (0.1, 0.2, 0.3)],
)
def test_add(x, y, expected):
    assert calculator.add(x, y) == pytest.approx(expected)


@pytest.mark.parametrize(
    "x, y, expected",
    [(2, 3, -1), (5, 0, 5), (-1, 1, -2), (-1, -1, 0)],
)
def test_subtract(x, y, expected):
    assert calculator.subtract(x, y) == expected


@pytest.mark.parametrize(
    "x, y, expected",
    [(2, 3, 6), (5, 0, 0), (-1, 1, -1), (-1, -1, 1)],
)
def test_multiply(x, y, expected):
    assert calculator.multiply(x, y) == expected


@pytest.mark.parametrize(
    "x, y, z, expected",
    [(2, 3, 5, 10), (5, 0, -1, 4), (-1, -1, -1, -3), (-1, -1, 100, 98)],
)
def test_add_three(x, y, z, expected):
    assert calculator.add_three(x, y, z) == expected


@pytest.mark.parametrize(
    "x, y, expected",
    [(6, 3, 2), (5, 2, 2.5), (-9, 3, -3), (0, 5, 0)],
)
def test_divide(x, y, expected):
    assert calculator.divide(x, y) == expected


def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        calculator.divide(1, 0)


@pytest.mark.parametrize(
    "base, exponent, expected",
    [(2, 3, 8), (5, 0, 1), (2, -1, 0.5), (-2, 2, 4)],
)
def test_power(base, exponent, expected):
    assert calculator.power(base, exponent) == expected


def test_power_zero_negative_exponent():
    with pytest.raises(ZeroDivisionError):
        calculator.power(0, -1)


@pytest.mark.parametrize(
    "values, expected",
    [([1, 2, 3], 2), ((4,), 4), ([-1, 1], 0), ([1.5, 2.5], 2)],
)
def test_mean(values, expected):
    assert calculator.mean(values) == expected


@pytest.mark.parametrize("empty", [[], ()])
def test_mean_rejects_empty(empty):
    with pytest.raises(ValueError):
        calculator.mean(empty)


@pytest.mark.parametrize("bad", ["123", None, [1, "2"]])
def test_mean_rejects_wrong_type(bad):
    with pytest.raises(TypeError):
        calculator.mean(bad)


@pytest.mark.parametrize(
    "func",
    [calculator.add, calculator.subtract, calculator.multiply,
     calculator.divide, calculator.power],
)
@pytest.mark.parametrize("bad", ["2", None, True, [1]])
def test_two_arg_functions_reject_non_numbers(func, bad):
    with pytest.raises(TypeError):
        func(bad, 1)
    with pytest.raises(TypeError):
        func(1, bad)


def test_add_three_rejects_non_numbers():
    with pytest.raises(TypeError):
        calculator.add_three(1, 2, "3")
