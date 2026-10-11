"""Simple calculator module used to demonstrate testing and CI/CD.

Changes from the original lab:
- Functions renamed from fun1..fun4 to descriptive names.
- Shared input validation (also rejects booleans, which Python treats as ints).
- Wrong input types now raise TypeError (Python convention) instead of ValueError.
- add_three() now validates its inputs like the other functions.
- New functions: divide(), power(), mean().
"""

from numbers import Real


def _validate(*values):
    """Raise TypeError unless every value is a real number (bools excluded)."""
    for v in values:
        if isinstance(v, bool) or not isinstance(v, Real):
            raise TypeError(f"Expected a number, got {v!r} ({type(v).__name__}).")


def add(x, y):
    """Return x + y."""
    _validate(x, y)
    return x + y


def subtract(x, y):
    """Return x - y."""
    _validate(x, y)
    return x - y


def multiply(x, y):
    """Return x * y."""
    _validate(x, y)
    return x * y


def add_three(x, y, z):
    """Return x + y + z (original fun4, now with validation)."""
    _validate(x, y, z)
    return x + y + z


def divide(x, y):
    """Return x / y. Raises ZeroDivisionError if y is 0."""
    _validate(x, y)
    if y == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return x / y


def power(base, exponent):
    """Return base ** exponent."""
    _validate(base, exponent)
    if base == 0 and exponent < 0:
        raise ZeroDivisionError("0 cannot be raised to a negative power.")
    return base**exponent


def mean(values):
    """Return the arithmetic mean of a non-empty list/tuple of numbers."""
    if not isinstance(values, (list, tuple)):
        raise TypeError("mean() needs a list or tuple of numbers.")
    if len(values) == 0:
        raise ValueError("mean() needs at least one number.")
    _validate(*values)
    return sum(values) / len(values)
