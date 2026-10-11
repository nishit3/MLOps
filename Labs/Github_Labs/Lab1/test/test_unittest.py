import unittest

from src import calculator


class TestCalculator(unittest.TestCase):

    def test_add(self):
        self.assertEqual(calculator.add(2, 3), 5)
        self.assertEqual(calculator.add(-1, -1), -2)
        self.assertAlmostEqual(calculator.add(0.1, 0.2), 0.3)

    def test_subtract(self):
        self.assertEqual(calculator.subtract(2, 3), -1)
        self.assertEqual(calculator.subtract(-1, -1), 0)

    def test_multiply(self):
        self.assertEqual(calculator.multiply(2, 3), 6)
        self.assertEqual(calculator.multiply(-1, -1), 1)

    def test_add_three(self):
        self.assertEqual(calculator.add_three(2, 3, 5), 10)
        self.assertEqual(calculator.add_three(-1, -1, 100), 98)

    def test_divide(self):
        self.assertEqual(calculator.divide(6, 3), 2)
        self.assertEqual(calculator.divide(5, 2), 2.5)
        with self.assertRaises(ZeroDivisionError):
            calculator.divide(1, 0)

    def test_power(self):
        self.assertEqual(calculator.power(2, 3), 8)
        self.assertEqual(calculator.power(2, -1), 0.5)
        with self.assertRaises(ZeroDivisionError):
            calculator.power(0, -1)

    def test_mean(self):
        self.assertEqual(calculator.mean([1, 2, 3]), 2)
        self.assertEqual(calculator.mean((4,)), 4)
        with self.assertRaises(ValueError):
            calculator.mean([])
        with self.assertRaises(TypeError):
            calculator.mean("123")

    def test_invalid_inputs(self):
        for bad in ["2", None, True, [1]]:
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    calculator.add(bad, 1)
                with self.assertRaises(TypeError):
                    calculator.add_three(1, 2, bad)


if __name__ == "__main__":
    unittest.main()
