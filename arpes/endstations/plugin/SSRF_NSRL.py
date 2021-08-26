# coding=utf-8
# Author : pfd@ruc.edu.cn, scw@ruc.edu.cn
"""
Implements data loading for the beamlines BL03U, BL09U (SSRF, China) 
and BL13U (NSRL, China).

In those beamlines (DA30-L Angle Resolved Electron Spectrometer), 
the format of file has been fixed :

    cut : '.pxt' ('.txt' sometimes. But don`t load '.text' file)
    map : '.zip'

There are the subfiles in '.zip' file (XXXX: sequence name):
    1.  XXXX.ini: header file 
    2.  Spectrum_XXXX.bin: Spectrum data
    3.  Spectrum_XXXX.ini: plugin
    4.  viewer.ini: plugin
    5.  viewer_settings.ini: plugin

"""
import io
from configparser import ConfigParser
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import xarray as xr
from arpes.endstations import SingleFileEndstation, SynchrotronEndstation
from arpes.load_pxt import read_single_pxt

__all__ = ('SSRFEndstation', 'NSRLEndstation')


class DA30_L(SingleFileEndstation):
    ALPHA = np.pi / 2

    PRINCIPAL_NAME = 'DA30' 
    ALIASES = ['DA30']             

    _SEARCH_DIRECTORIES = ('zip', 'pxt')        
    _TOLERATED_EXTENSIONS = {'.zip', '.pxt'}    

    ENSURE_COORDS_EXIST = [
        "hv", "alpha", "psi", 
        "theta", "beta", "chi", # convert kspcae need them
        ] 
        
    RENAME_KEYS = {
        'sample': 'sample_name',
        'uesr' : 'experimenter',
        'spectrum_name' : 'spectrum_type',
        'low_energy' : 'sweep_low_energy',
        'center_energy' : 'sweep_center_energy',
        'high_energy' : 'sweep_high_energy',
        'step_time' : 'n_sweeps',
        'energy_step' : 'sweep_step',
        'instrument' : 'analyzer',
        'region_name': 'spectrum_type',
    }

    MERGE_ATTRS = {
        "analyzer_name": "DA30-L",
        'analyzer_type': 'hemispherical',
        'detect_radius': '15 degrees',
        'perpendicular_deflectors': 'True',
        'alpha': ALPHA,
    }

    def load_single_frame(self, fpath: str = None, scan_desc: dict = None, **kwargs):
        file = Path(fpath)

        if file.suffix == '.pxt':
            frame = read_single_pxt(fpath).rename(W = 'eV', X = 'phi')
            frame.assign_coords(phi = np.deg2rad(frame.phi))
            
            return xr.Dataset(
                {'spectrum': frame},
                attrs=scan_desc,
            )

        if file.suffix == '.zip':
            zf = ZipFile(fpath)
            viewer_ini_ziped = zf.open('viewer.ini', 'r')
            viewer_ini_io = io.TextIOWrapper(viewer_ini_ziped)
            viewer_ini = ConfigParser(strict = False)
            viewer_ini.read_file(viewer_ini_io)                    
                         
            def determine_dim(dim_name):
                spectrum_info = viewer_ini.sections()[-1]

                num = viewer_ini.getint(spectrum_info, dim_name)

                offset = viewer_ini.getfloat(spectrum_info, dim_name + '_offset') 
                delta = viewer_ini.getfloat(spectrum_info, dim_name + '_delta')            
                end = offset + num * delta
                coord = np.linspace(offset, end, num = num, endpoint= False)

                name = viewer_ini.get(spectrum_info, dim_name + '_label') 
                return num, coord, name

    
            # Usually, ['width', 'height', 'depth'] -> ['eV', 'phi', 'psi']
            # For safety, get its label name and sort it
            raw_coords = {}
            for label in ['width', 'height', 'depth']:
                num, data, name = determine_dim(label)
                raw_coords[name] = [num, data]
            raw_coords_name = list(raw_coords.keys())
            raw_coords_name.sort() 
    
            # After sorting, labels must be ['Energy [eV]', 'Thetax [deg]', 
            # 'Thetay [deg]'].Those means ['eV', 'phi', 'psi'].
            built_coords = {
                'psi': raw_coords[raw_coords_name[2]][1],
                'phi': raw_coords[raw_coords_name[1]][1],
                'eV': raw_coords[raw_coords_name[0]][1],
            } 
            (psi_num, phi_num, eV_num) = (
                raw_coords[raw_coords_name[2]][0],
                raw_coords[raw_coords_name[1]][0],
                raw_coords[raw_coords_name[0]][0],
            )
        
            data_path = viewer_ini.get(viewer_ini.sections()[-1], 'path')
            raw_data = zf.read(data_path)
            loaded_data = np.frombuffer(raw_data, dtype='float32')
            loaded_data.shape = (psi_num, phi_num, eV_num)

            attr_path = viewer_ini.get(viewer_ini.sections()[0], 'ini_path')
            
            attr_ziped = zf.open(attr_path, 'r')
            attr_io = io.TextIOWrapper(attr_ziped)
            attr_conf = ConfigParser(strict = False)
            attr_conf.read_file(attr_io)
            
            attrs = {
                v.replace(' ', '_'): k 
                for section in attr_conf.sections() 
                for v,k in attr_conf.items(section)
            }
            data = xr.DataArray(
                loaded_data,
                dims = ['psi', 'phi', 'eV'], 
                coords = built_coords,
                attrs = attrs, 
            )
            data.assign_coords(
                phi = np.deg2rad(data.phi),
                psi = np.deg2rad(data.psi),
            )
       
            return xr.Dataset(
                {'spectrum': data},
                attrs={**scan_desc},
            )


class SSRFEndstation(DA30_L, SynchrotronEndstation):
    # BL03U and BL09U at the Shanghai Synchrotron Radiation Facility 
    # (SSRF, China)
    PRINCIPAL_NAME = 'SSRF' 
    ALIASES = [
        'SSRF',
        'SSRF-BL03U',
        'SSRF-BL09U',
        'BL03U',
        'BL09U',
        'Dearmline',
    ] 


class NSRLEndstation(DA30_L, SynchrotronEndstation):
    # BL13U at the National Synchrotron Radiation Laboratory 
    # (NSRL, China)
    PRINCIPAL_NAME = 'NSRL' 
    ALIASES = [
        'NSRL',
        'NSRL-BL13U',
        'BL13U',
    ] 


