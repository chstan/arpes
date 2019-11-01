import warnings

from arpes.endstations import EndstationBase, resolve_endstation

__all__ = ('FallbackEndstation',)

AUTOLOAD_WARNING = (
    'PyARPES has chosen {} for your data since `location` was not specified. '
    'If this is not correct, ensure the `location` key is specified. Read the plugin documentation '
    'for more details.'
)


class FallbackEndstation(EndstationBase):
    """
    Different from the rest of the data loaders. This one is used when there is no location specified
    and attempts sequentially to call a variety of standard plugins until one is found that works.
    """
    PRINCIPAL_NAME = 'fallback'
    ALIASES = []

    ATTEMPT_ORDER = [
        'ANTARES',
        'MBS',
        'ALS-BL7',
        'ALS-BL403',
        'Kaindl',
        'ALG-Main',
        'ALG-SToF',
    ]

    @classmethod
    def determine_associated_loader(cls, file, scan_desc):
        import arpes.config  # pylint: disable=redefined-outer-name
        arpes.config.load_plugins()

        for location in cls.ATTEMPT_ORDER:
            try:
                endstation_cls = resolve_endstation(False, location=location)
                if endstation_cls.is_file_accepted(file, scan_desc):
                    return endstation_cls
            except:
                pass

        raise ValueError(f'PyARPES failed to find a plugin acceptable for {file}, \n\n{scan_desc}.')

    def load(self, scan_desc: dict = None, file=None, **kwargs):
        if file is None:
            file = scan_desc['file']

        associated_loader = FallbackEndstation.determine_associated_loader(file, scan_desc)

        try:
            file = int(file)
            file = associated_loader.find_first_file(file, scan_desc)
            scan_desc['file'] = file
        except ValueError:
            pass

        warnings.warn(AUTOLOAD_WARNING.format(associated_loader))
        return associated_loader().load(scan_desc, **kwargs)

    @classmethod
    def find_first_file(cls, file, scan_desc, allow_soft_match=False):
        associated_loader = cls.determine_associated_loader(file, scan_desc)
        warnings.warn(AUTOLOAD_WARNING.format(associated_loader))
        return associated_loader.find_first_file(file, scan_desc, allow_soft_match)

