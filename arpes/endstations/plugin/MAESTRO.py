from arpes.endstations import HemisphericalEndstation, SynchrotronEndstation, FITSEndstation

__all__ = ('MAESTROARPESEndstation',)


class MAESTROARPESEndstation(SynchrotronEndstation, HemisphericalEndstation, FITSEndstation):
    """
    The MERLIN ARPES Endstation at the Advanced Light Source
    """

    PRINCIPAL_NAME = 'ALS-BL702'
    ALIASES = ['BL7', 'BL7.0.2', 'ALS-BL7.0.2',]

    RENAME_KEYS = {
        'LMOTOR0': 'x',
        'LMOTOR1': 'y',
        'LMOTOR2': 'z',
        'LMOTOR3': 'theta',
        'LMOTOR4': 'beta',
        'LMOTOR5': 'sample-phi',
        'LMOTOR9': 'Slit Defl',
    }