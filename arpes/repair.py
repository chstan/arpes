from arpes.typing import DataType

__all__ = ('negate_energy',)

def negate_energy(data: DataType):
    if 'eV' in data.coords:
        data = data.assign_coords(eV=-data.eV.values)
        
    return data
