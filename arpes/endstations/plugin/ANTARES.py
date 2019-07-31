import xarray as xr
import numpy as np
import h5py

from arpes.endstations import HemisphericalEndstation, SynchrotronEndstation, SingleFileEndstation
from arpes.endstations.nexus_utils import read_data_attributes_from
from arpes.preparation import disambiguate_coordinates

__all__ = ('ANTARESEndstation',)


mono = [['ANTARES', 'Monochromator'],
        ['exitSlitAperture', 'resolution', 'currentGratingName', 'currentSlotName', 'energy']]
user_info = [['User'],
             ['email', 'address', 'affiliation', 'name', 'telephone_number']]
misc = [[], ['comment_conditions', 'experimental_frame', 'start_time']]


general_paths = [mono, user_info, misc]

# not generally needed
mbs_general_location = [[], ['PI-X', 'PI-Y', 'PI-Z', 'Phi', 'Theta']]
mbs_general_paths = [mbs_general_location]

# To be run inside ANTARES/MBSAcquisition_{idx}
mbs_acquisition_spectrometer = [[], ['Frames', 'LensMode', 'PASSENERGY', 'DeflX', 'DeflY', 'CenterKE', 'StepSize',
                                     'StartX', 'StartY', 'EndX', 'EndY', 'StartKE', 'NoSlices', 'NoScans']]
mbs_acquisition_paths = [mbs_acquisition_spectrometer]


def parse_axis_name_from_long_name(name):
    return name.split('/')[-1].replace("'", '')


def infer_scan_type_from_data(group):
    """
    Because ANTARES stores every possible data type in the NeXuS file format, zeroing information that is
    not used, we have to determine which data folder to use on the basis of what kind of scan was done.
    """
    scan_name = str(group['scan_config']['name'].value)

    if 'DeflX' in scan_name:
        # Fermi Surface, might need to be more robust
        return 'data_09'

    if 'Scan2D_MBS' in scan_name:
        # two piezo or two DOF image scan
        return 'data_12'

    raise NotImplementedError(scan_name)


class ANTARESEndstation(HemisphericalEndstation, SynchrotronEndstation, SingleFileEndstation):
    """
    Implements loading text files from the MB Scientific text file format.

    There's not too much metadata here except what comes with the analyzer settings.
    """

    PRINCIPAL_NAME = 'ANTARES'
    ALIASES = []

    RENAME_KEYS = {
        'deflx': 'psi',
    }

    def load_top_level_scan(self, group, scan_desc: dict=None, spectrum_index=None):
        dr = self.read_scan_data(group)
        attrs = read_data_attributes_from(group, general_paths)

        try:
            mbs_key = [k for k in list(group['ANTARES'].keys()) if 'MBSAcquisition' in k][0]
            attrs.update(read_data_attributes_from(group['ANTARES'][mbs_key], mbs_acquisition_paths))
        except IndexError:
            pass

        dr = dr.assign_attrs(attrs)
        return xr.Dataset(dict([['spectrum-{}'.format(spectrum_index), dr]]))

    def get_coords(self, group, scan_name, shape):
        # This will have to be modified for data which lacks either a phi or energy axis
        # We will cross this bridge once we have any idea what shape the bridge is in

        dims = list(shape)
        data = group['scan_data']

        # handle actuators
        relaxed_shape = list(shape)
        actuator_list = [k for k in list(data.keys()) if 'actuator' in k]
        actuator_names = [parse_axis_name_from_long_name(
            str(data[act].attrs['long_name'])) for act in actuator_list]
        actuator_list = [data[act][:] for act in actuator_list]

        actuator_dim_order = []
        for act in actuator_list:
            found = relaxed_shape.index(act.shape[-1])
            actuator_dim_order.append(found)
            relaxed_shape[found] = None

        coords = {}

        def take_last(vs):
            while len(vs.shape) > 1:
                vs = vs[0]

            return vs

        for dim_order, name, values in zip(actuator_dim_order, actuator_names, actuator_list):
            name = self.RENAME_KEYS.get(name, name)
            dims[dim_order] = name
            coords[name] = take_last(values)

        # handle standard spectrometer axes, keeping in mind things get stored
        # in different places sometimes for no reasons
        energy_keys = {
            'data_9': ('data_01', 'data_03', 'data_02',),
            'data_12': ('data_04', 'data_06', 'data_05',),
        }
        angle_keys = {
            'data_9': ('data_04', 'data_06', 'data_05',),
            'data_12': ('data_07', 'data_09', 'data_08',),
        }
        e_keys = energy_keys[scan_name]
        ang_keys = angle_keys[scan_name]
        energy = data[e_keys[0]][0], data[e_keys[1]][0], data[e_keys[2]][0]
        angle = data[ang_keys[0]][0], data[ang_keys[1]][0], data[ang_keys[2]][0]

        def get_first(item):
            if isinstance(item, np.ndarray):
                return item.ravel()[0]

            return item

        def build_axis(low, high, step_size):
            # this might not work out to be the right thing to do, we will see
            low, high, step_size = get_first(low), get_first(high), get_first(step_size)
            est_n = int((high - low) / step_size)

            closest = None
            diff = np.inf
            idx = None
            for i, s in enumerate(shape):
                if closest is None or np.abs(s - est_n) < diff:
                    idx = i
                    diff = np.abs(s - est_n)
                    closest = s

            return np.linspace(low, high, closest, endpoint=False), idx

        energy, energy_idx = build_axis(*energy)
        angle, angle_idx = build_axis(*angle)

        dims[energy_idx] = 'eV'
        dims[angle_idx] = 'phi'
        coords['eV'] = energy
        coords['phi'] = angle * np.pi / 180

        return dims, coords

    def read_scan_data(self, group):
        """
        Reads the scan data stored in /scan_data/data_{idx} as appropriate for the type of file.
        """
        data_key = infer_scan_type_from_data(group)
        data_group = group['scan_data'][data_key]
        data = data_group[:]

        dims, coords = self.get_coords(group, data_key, shape=data.shape)

        return xr.DataArray(data, coords=coords, dims=dims)

    def load_single_frame(self, frame_path: str=None, scan_desc: dict=None, **kwargs):
        f = h5py.File(frame_path)
        top_level = list(f.keys())

        loaded = [self.load_top_level_scan(f[key], scan_desc, spectrum_index=i) for i, key in enumerate(top_level)]

        if isinstance(loaded, list) and len(loaded) > 0:
            loaded = disambiguate_coordinates(loaded, ['phi', 'eV'])
            loaded = xr.merge(loaded)
        else:
            loaded = loaded[0]
            loaded.rename({'spectrum-1': 'spectrum'})

        return loaded
