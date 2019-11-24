from arpes.io import load_example_data
from arpes.plotting.dynamic_tool import make_dynamic

data = load_example_data()


def adjust_gamma(data, gamma: float = 1):
    """
    Equivalent to adjusting the display gamma factor, just rescale the data
    according to f(x) = x^gamma.

    :param data:
    :param gamma:
    :return:
    """
    return data ** gamma


make_dynamic(adjust_gamma, data)

