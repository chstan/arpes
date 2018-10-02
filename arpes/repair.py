from arpes.typing import DataType

__all__ = ('negate_energy',)

def negate_energy(data: DataType):
    if 'eV' in data.coords:
        data.coords['eV'].values = -data.coords['eV'].values
    return data