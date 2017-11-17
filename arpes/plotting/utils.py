import datetime
import errno
import itertools
import os.path
from collections import Counter

from arpes.config import CONFIG, FIGURE_PATH

__all__ = ['path_for_plot', 'path_for_holoviews', 'name_for_dim', 'label_for_colorbar', 'label_for_dim',
           'label_for_symmetry_point']


def path_for_plot(desired_path):
    workspace = CONFIG['WORKSPACE']
    assert(workspace is not None)

    filename = os.path.join(FIGURE_PATH, workspace,
                            datetime.date.today().isoformat(), desired_path)
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    return filename


def path_for_holoviews(desired_path):
    skip_paths = ['.svg', '.png', '.jpeg', '.jpg', '.gif']

    prefix, ext = os.path.splitext(desired_path)

    if ext in skip_paths:
        return prefix

    return prefix + ext


def name_for_dim(dim_name, escaped=True):
    name = {
        'polar': r'$\theta$',
        'phi': r'$\varphi$',
        'eV': r'$\textnormal{E}$',
        'kx': r'$\textnormal{k}_\textnormal{x}$',
        'ky': r'$\textnormal{k}_\textnormal{y}$',
        'kz': r'$\textnormal{k}_\textnormal{z}$',
        'kp': r'$\textnormal{k}_\textnormal{\parallel}$',
    }.get(dim_name)

    if not escaped:
        name = name.replace('$', '')

    return name


def label_for_colorbar(data):
    if not data.S.is_differentiated:
        return r'Spectrum Intensity (arb).'

    # determine which axis was differentiated
    hist = data.S.history
    records = [h['record'] for h in hist if isinstance(h, dict)]
    if 'curvature' in [r['by'] for r in records]:
        curvature_record = [r for r in records if r['by'] == 'curvature'][0]
        directions = curvature_record['directions']
        return r'Curvature along {} and {}'.format(
            name_for_dim(directions[0]),
            name_for_dim(directions[1])
        )

    derivative_records = [r for r in records if r['by'] == 'dn_along_axis']
    c = Counter(itertools.chain(*[[d['axis']] * d['order'] for d in derivative_records]))

    partial_frag = r''
    if sum(c.values()) > 1:
        partial_frag = r'^' + str(sum(c.values()))

    return r'$\frac{\partial' + partial_frag + r' \textnormal{Int.}}{' + \
           r''.join([r'\partial {}^{}'.format(name_for_dim(item, escaped=False), n)
                     for item, n in c.items()])+ '}$ (arb.)'


def label_for_dim(data, dim_name, escaped=True):
    raw_dim_names = {
        'polar': r'$\theta$',
        'phi': r'$\varphi$',
        'eV': r'\textbf{eV}'
    }

    if data.S.spectrometer.get('type') == 'hemisphere':
        raw_dim_names['phi'] = r'$\varphi$ (Hemisphere Acceptance)'

    if dim_name in raw_dim_names:
        return raw_dim_names.get(dim_name)

    # Next we will look at the listed symmetry_points to try to infer the appropriate way to display the axis
    return r'$\boldsymbol{\Gamma}\rightarrow \textbf{X}$'


def label_for_symmetry_point(point_name):
    proper_names = {
        'G': r'$\Gamma$',
        'X': r'X',
        'Y': r'Y',
    }
    return proper_names.get(point_name, point_name)