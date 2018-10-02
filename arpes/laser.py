from arpes.config import ureg

__all__ = ('electrons_per_pulse',)

mira_frequency = (54.3 / ureg.microsecond)


def electrons_per_pulse(photocurrent, division_ratio=None, pulse_rate=None):
    assert(division_ratio is None or pulse_rate is None)

    if division_ratio is not None:
        pulse_rate = mira_frequency / division_ratio

    eles_per_attocoulomb = 6.2415091
    atto_coulombs = (photocurrent / pulse_rate).to('attocoulomb')

    return (atto_coulombs * eles_per_attocoulomb).magnitude