import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import itertools
import matplotlib.pyplot as plt
from functools import wraps
from matplotlib.widgets import LassoSelector, Button, TextBox, RectangleSelector, SpanSelector, Slider
from matplotlib.path import Path

from arpes.plotting.utils import imshow_arr, fancy_labels, invisible_axes
from arpes.fits import LorentzianModel, broadcast_model
from arpes.utilities import normalize_to_spectrum
from arpes.utilities.conversion import convert_to_kspace

__all__ = ('pick_rectangles', 'pick_points', 'pca_explorer', 'kspace_tool',)


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Modified from https://matplotlib.org/gallery/widgets/lasso_selector_demo_sgskip.html

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3, on_select=None):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)
        self._on_select = on_select

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        try:
            path = Path(verts)
            self.ind = np.nonzero(path.contains_points(self.xys))[0]
            self.fc[:, -1] = self.alpha_other
            self.fc[self.ind, -1] = 1
            self.collection.set_facecolors(self.fc)
            self.canvas.draw_idle()

            if self._on_select is not None:
                self._on_select(self.ind)
        except Exception:
            pass

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def popout(plotting_function):
    """
    Sets and subsequently unsets the matplotlib backend for one function call, to allow use of
    'widgets' in Jupyter inline use.

    :param plotting_function:
    :return:
    """

    @wraps(plotting_function)
    def wrapped(*args, **kwargs):
        from IPython import get_ipython

        ipython = get_ipython()
        ipython.magic('matplotlib qt')

        return plotting_function(*args, **kwargs)

        # ideally, cleanup, but this closes the plot, necessary but redundant looking import
        # look into an on close event for matplotlib
        #ipython.magic('matplotlib inline')
        #from matplotlib import pyplot as plt

    return wrapped


class DataArrayView(object):
    """
    Offers support for 1D and 2D DataArrays with masks, selection tools, and a simpler interface
    than the matplotlib primitives.

    Look some more into holoviews for different features. https://github.com/pyviz/holoviews/pull/1214
    """
    def __init__(self, ax, data=None, ax_kwargs=None, mask_kwargs=None, transpose_mask=False, auto_autoscale=True):
        self.ax = ax
        self._initialized = False
        self._data = None
        self._mask = None
        self.n_dims = None
        self.ax_kwargs = ax_kwargs or dict()
        self._axis_image = None
        self._mask_image = None
        self._mask_cmap = None
        self._transpose_mask = transpose_mask
        self._selector = None
        self._inner_on_select = None
        self.auto_autoscale = auto_autoscale
        self.mask_kwargs = mask_kwargs

        if data is not None:
            self.data = data

    def handle_select(self, event_click=None, event_release=None):
        dims = self.data.dims

        if self.n_dims == 2:
            x1, y1 = event_click.xdata, event_click.ydata
            x2, y2 = event_release.xdata, event_release.ydata

            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            region = dict([[dims[1], slice(x1, x2)], [dims[0], slice(y1, y2)]])
        else:
            x1, x2 = event_click, event_release
            x1, x2 = min(x1, x2), max(x1, x2)

            region = dict([[self.data.dims[0], slice(x1, x2)]])

        self._inner_on_select(region)

    def attach_selector(self, on_select):
        # data should already have been set
        assert(self.n_dims is not None)

        self._inner_on_select = on_select

        if self.n_dims == 1:
            self._selector = SpanSelector(
                self.ax, self.handle_select, 'horizontal',
                useblit=True, rectprops=dict(alpha=0.35, facecolor='red'),
            )
        else:
            self._selector = RectangleSelector(
                self.ax, self.handle_select, drawtype='box',
                rectprops=dict(fill=False, edgecolor='black', linewidth=2),
                lineprops=dict(linewidth=2, color='black'),
            )

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        if self._initialized:
            self._data = new_data
        else:
            self._data = new_data
            self._initialized = True
            self.n_dims = len(new_data.dims)
            if self.n_dims == 2:
                self._axis_image = imshow_arr(self._data, ax=self.ax, **self.ax_kwargs)[1]
                fancy_labels(self.ax)
            else:
                self.ax_kwargs.pop('cmap', None)
                x, y = self.data.coords[self.data.dims[0]].values, self.data.values
                self._axis_image = self.ax.plot(x, y, **self.ax_kwargs)
                self.ax.set_xlabel(self.data.dims[0])
                cs = self.data.coords[self.data.dims[0]].values
                self.ax.set_xlim([np.min(cs), np.max(cs)])
                fancy_labels(self.ax)

        if self.n_dims == 2:
            x, y = self._data.coords[self._data.dims[0]].values, self._data.coords[self._data.dims[1]].values
            extent = [y[0], y[-1], x[0], x[-1]]
            self._axis_image.set_extent(extent)
            self._axis_image.set_data(self._data.values)
        else:
            color = self.ax.lines[0].get_color()
            self.ax.lines.remove(self.ax.lines[0])
            x, y = self.data.coords[self.data.dims[0]].values, self.data.values
            l, h = np.min(y), np.max(y)
            self._axis_image = self.ax.plot(x, y, c=color, **self.ax_kwargs)
            self.ax.set_ylim([l - 0.1 * (h - l), h + 0.1 * (h - l)])

        if self.auto_autoscale:
            self.autoscale()

    @property
    def mask_cmap(self):
        if self._mask_cmap is None:
            self._mask_cmap = matplotlib.cm.get_cmap(self.mask_kwargs.pop('cmap', 'Reds'))
            self._mask_cmap.set_bad('k', alpha=0)

        return self._mask_cmap

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, new_mask):
        if np.array(new_mask).shape != self.data.values.shape:
            # should be indices then
            mask = np.zeros(self.data.values.shape, dtype=bool)
            np.ravel(mask)[new_mask] = True
            new_mask = mask

        self._mask = new_mask

        for_mask = np.ma.masked_where(np.logical_not(self._mask), self.data.values * 0 + 1)
        if self.n_dims == 2 and self._transpose_mask:
            for_mask = for_mask.T

        if self.n_dims == 2:
            if self._mask_image is None:
                self._mask_image = self.ax.imshow(
                    for_mask.T, cmap=self.mask_cmap, interpolation='none', vmax=1, vmin=0,
                    origin='lower', extent=self._axis_image.get_extent(),
                    aspect=self.ax.get_aspect(), **self.mask_kwargs
                )
            else:
                self._mask_image.set_data(for_mask.T)
        else:
            if self._mask_image is not None:
                self.ax.collections.remove(self._mask_image)

            x = self.data.coords[self.data.dims[0]].values
            low, high = self.ax.get_ylim()
            self._mask_image = self.ax.fill_between(x, low, for_mask * high, color=self.mask_cmap(1.), **self.mask_kwargs)

    def autoscale(self):
        if self.n_dims == 2:
            self._axis_image.autoscale()
        else:
            pass


@popout
def fit_initializer(data, peak_type=LorentzianModel, **kwargs):
    ctx = {}
    gs = gridspec.GridSpec(2, 2)
    ax_initial = plt.subplot(gs[0, 0])
    ax_fitted = plt.subplot(gs[0, 1])
    ax_other = plt.subplot(gs[1, 0])
    ax_test = plt.subplot(gs[1, 1])

    invisible_axes(ax_other)

    prefixes = 'abcdefghijklmnopqrstuvwxyz'
    model_settings = []
    model_defs = []
    fitted_individual_models = []
    for_fit = data.expand_dims('fit_dim')
    for_fit.coords['fit_dim'] = np.array([0])

    data_view = DataArrayView(ax_initial)
    residual_view = DataArrayView(ax_fitted, ax_kwargs=dict(linestyle=':', color='orange'))
    fitted_view = DataArrayView(ax_fitted, ax_kwargs=dict(color='red'))
    initial_fit_view = DataArrayView(ax_fitted, ax_kwargs=dict(linestyle='--', color='blue'))

    def compute_parameters():
        renamed = [{'{}_{}'.format(prefix, k): v for k, v in m_setting.items()} for m_setting, prefix in zip(model_settings, prefixes)]
        return dict(itertools.chain(*[list(d.items()) for d in renamed]))

    def on_add_new_peak(selection):
        amplitude = data.sel(**selection).mean().item()

        selection = selection[data.dims[0]]
        center = (selection.start + selection.stop) / 2
        sigma = (selection.stop - selection.start)

        model_settings.append({'center': {'value': center, 'min': center - sigma, 'max': center + sigma}, 'sigma': {'value': sigma}, 'amplitude': {'min': 0, 'value': amplitude}})
        model_defs.append(LorentzianModel)

        if len(model_defs):
            results = broadcast_model(model_defs, for_fit, 'fit_dim', params=compute_parameters())
            result = results.results[0].item()

            if result is not None:
                # residual
                for_residual = data.copy(deep=True)
                for_residual.values = result.residual
                residual_view.data = for_residual

                # fit_result
                for_best_fit = data.copy(deep=True)
                for_best_fit.values = result.best_fit
                fitted_view.data = for_best_fit

                # initial_fit_result
                for_initial_fit = data.copy(deep=True)
                for_initial_fit.values = result.init_fit
                initial_fit_view.data = for_initial_fit

                ax_fitted.set_ylim(ax_initial.get_ylim())


    data_view.data = data
    data_view.attach_selector(on_select=on_add_new_peak)
    ctx['data'] = data

    def on_copy_settings(event):
        try:
            import pyperclip
            import pprint
            pyperclip.copy(pprint.pformat(compute_parameters()))
        except ImportError:
            pass
        finally:
            import pprint
            print(pprint.pformat(compute_parameters()))

    copy_settings_button = Button(ax_test, 'Copy Settings')
    copy_settings_button.on_clicked(on_copy_settings)
    ctx['button'] = copy_settings_button
    return ctx


@popout
def pca_explorer(pca, data, component_dim='components', initial_values=None, transpose_mask=False, **kwargs):
    if initial_values is None:
        initial_values = [0, 1]

    pca_dims = list(pca.dims)
    pca_dims.remove(component_dim)
    other_dims = [d for d in data.dims if d not in pca_dims]

    context = {
        'selected_components': initial_values,
        'selected_indices': [],
        'sum_data': None,
        'map_data': None,
        'selector': None,
        'integration_region': None,
    }

    def compute_for_scatter():
        for_scatter = pca.copy(deep=True).isel(**dict([[component_dim, context['selected_components']]]))
        for_scatter = for_scatter.S.transpose_to_back(component_dim)

        size = data.mean(other_dims).stack(pca_dims=pca_dims).values
        norm = np.expand_dims(np.linalg.norm(pca.values, axis=(0,)), axis=-1)

        return (for_scatter / norm).stack(pca_dims=pca_dims), 5 * size / np.mean(size)

    # ===== Set up axes ======
    gs = gridspec.GridSpec(2, 2)
    ax_components = plt.subplot(gs[0, 0])
    ax_sum_selected = plt.subplot(gs[0, 1])
    ax_map = plt.subplot(gs[1, 0])

    gs_widget = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1])
    ax_widget_1 = plt.subplot(gs_widget[0, 0])
    ax_widget_2 = plt.subplot(gs_widget[1, 0])
    ax_widget_3 = plt.subplot(gs_widget[2, 0])

    selected_view = DataArrayView(ax_sum_selected, ax_kwargs=dict(cmap='viridis'))
    map_view = DataArrayView(ax_map, ax_kwargs=dict(cmap='Greys'),
                             mask_kwargs=dict(cmap='Reds', alpha=0.35), transpose_mask=transpose_mask)

    def update_from_selection(ind):
        # Calculate the new data
        if ind is None or len(ind) == 0:
            context['selected_indices'] = []
            context['sum_data'] = data.stack(pca_dims=pca_dims).sum('pca_dims')
        else:
            context['selected_indices'] = ind
            context['sum_data'] = data.stack(pca_dims=pca_dims).isel(pca_dims=ind).sum('pca_dims')

        if context['integration_region'] is not None:
            data_sel = data.sel(**context['integration_region']).sum(other_dims)
        else:
            data_sel = data.sum(other_dims)

        # Update all views
        map_view.data = data_sel
        map_view.mask = ind
        selected_view.data = context['sum_data']

    def set_axes(component_x, component_y):
        ax_components.clear()
        context['selected_components'] = [component_x, component_y]
        for_scatter, size = compute_for_scatter()
        pts = ax_components.scatter(for_scatter.values[0], for_scatter.values[1], s=size)

        if context['selector'] != None:
            context['selector'].disconnect()

        context['selector'] = SelectFromCollection(ax_components, pts, on_select=update_from_selection)
        ax_components.set_xlabel('$e_' + str(component_x) + '$')
        ax_components.set_ylabel('$e_' + str(component_y) + '$')
        update_from_selection([])

    def on_change_axes(event):
        try:
            val_x = int(context['axis_X_input'].text)
            val_y = int(context['axis_Y_input'].text)

            def clamp(x, low, high):
                if low <= x < high:
                    return x
                if x < low:
                    return low
                return high

            maximum = len(pca.coords[component_dim].values) - 1

            val_x, val_y = clamp(val_x, 0, maximum), clamp(val_y, 0, maximum)

            assert(val_x != val_y)

            set_axes(val_x, val_y)
        except Exception as e:
            pass

    context['axis_button'] = Button(ax_widget_1, 'Change Decomp Axes')
    context['axis_button'].on_clicked(on_change_axes)
    context['axis_X_input'] = TextBox(ax_widget_2, 'Axis X:', initial=str(initial_values[0]))
    context['axis_Y_input'] = TextBox(ax_widget_3, 'Axis Y:', initial=str(initial_values[1]))

    def on_select_summed(region):
        context['integration_region'] = region
        update_from_selection(context['selected_indices'])

    set_axes(*initial_values)
    selected_view.attach_selector(on_select_summed)

    plt.tight_layout()
    return context


@popout
def kspace_tool(data, **kwargs):
    original_data = data
    data = normalize_to_spectrum(data)
    if len(data.dims) > 2:
        data = data.sel(eV=slice(-0.05, 0.05)).sum('eV', keep_attrs=True)
        data.coords['eV'] = 0

    if 'eV' in data.dims:
        data = data.S.transpose_to_front('eV')

    data = data.copy(deep=True)

    ctx = {
        'original_data': original_data,
        'data': data,
        'widgets': []
    }
    gs = gridspec.GridSpec(4, 3)
    ax_initial = plt.subplot(gs[0:2, 0:2])
    ax_converted = plt.subplot(gs[2:, 0:2])

    n_widget_axes = 8
    gs_widget = gridspec.GridSpecFromSubplotSpec(n_widget_axes, 1, subplot_spec=gs[:, 2])

    widget_axes = [plt.subplot(gs_widget[i, 0]) for i in range(n_widget_axes)]
    [invisible_axes(a) for a in widget_axes[:-2]]

    skip_dims = {'x', 'X', 'y', 'y', 'z', 'Z', 'T'}
    for dim in skip_dims:
        if dim in data.dims:
            raise ValueError('Please provide data without the {} dimension'.format(dim))

    convert_dims = ['theta', 'beta', 'phi', 'psi']
    if 'eV' not in data.dims:
        convert_dims += ['chi']
    if 'hv' in data.dims:
        convert_dims += ['hv']

    ang_range = (-45 * np.pi / 180, 45 * np.pi / 180, 0.01)
    default_ranges = {
        'eV': [-0.05, 0.05, 0.001],
        'hv': [-20, 20, 0.5],
    }

    sliders = {}

    def update_kspace_plot(_):
        for name, slider in sliders.items():
            data.attrs['{}_offset'.format(name)] = slider.val

        converted_view.data = convert_to_kspace(data)

    axes = iter(widget_axes)
    for convert_dim in convert_dims:
        widget_ax = next(axes)
        low, high, delta = default_ranges.get(convert_dim, ang_range)
        init = data.S._lookup_offset(convert_dim)
        sliders[convert_dim] = Slider(widget_ax, convert_dim, init + low, init + high, valinit=init, valstep=delta)
        sliders[convert_dim].on_changed(update_kspace_plot)

    def compute_offsets():
        return {k: v.val for k, v in sliders.items()}

    def on_copy_settings(event):
        try:
            import pyperclip
            import pprint
            pyperclip.copy(pprint.pformat(compute_offsets()))
        except ImportError:
            pass
        finally:
            import pprint
            print(pprint.pformat(compute_offsets()))

    def apply_offsets(event):
        for name, offset in compute_offsets().items():
            print(name, offset)
            original_data.attrs['{}_offset'.format(name)] = offset
            try:
                for s in original_data.S.spectra:
                    s.attrs['{}_offset'.format(name)] = offset
            except AttributeError:
                pass

    ctx['widgets'].append(sliders)

    copy_settings_button = Button(widget_axes[-1], 'Copy Offsets')
    apply_settings_button = Button(widget_axes[-2], 'Apply Offsets')
    copy_settings_button.on_clicked(on_copy_settings)
    apply_settings_button.on_clicked(apply_offsets)
    ctx['widgets'].append(copy_settings_button)
    ctx['widgets'].append(apply_settings_button)

    data_view = DataArrayView(ax_initial)
    converted_view = DataArrayView(ax_converted)

    data_view.data = data
    update_kspace_plot(None)

    plt.tight_layout()

    return ctx


@popout
def pick_rectangles(data, **kwargs):
    ctx = {'points': [], 'rect_next': False}
    rects = []

    fig = plt.figure()
    data.S.plot(**kwargs)
    ax = fig.gca()

    def onclick(event):
        ctx['points'].append([event.xdata, event.ydata])
        if ctx['rect_next']:
            p1, p2 = ctx['points'][-2], ctx['points'][-1]
            p1[0], p2[0] = min(p1[0], p2[0]), max(p1[0], p2[0])
            p1[1], p2[1] = min(p1[1], p2[1]), max(p1[1], p2[1])

            rects.append([p1, p2])
            rect = plt.Rectangle((p1[0], p1[1],), p2[0] - p1[0], p2[1] - p1[1],
                                 edgecolor='red', linewidth=2, fill=False)
            ax.add_patch(rect)

        ctx['rect_next'] = not ctx['rect_next']
        plt.draw()

    cid = plt.connect('button_press_event', onclick)

    return rects


@popout
def pick_gamma(data, **kwargs):
    fig = plt.figure()
    data.S.plot(**kwargs)

    ax = fig.gca()
    dims = data.dims

    def onclick(event):

        data.attrs['symmetry_points'] = {
            'G': {}
        }

        print(event.x, event.xdata, event.y, event.ydata)

        for dim, value in zip(dims, [event.ydata, event.xdata]):
            if dim == 'eV':
                continue

            data.attrs['symmetry_points']['G'][dim] = value

        plt.draw()

    cid = plt.connect('button_press_event', onclick)

    return data


@popout
def pick_points(data, **kwargs):
    ctx = {'points': []}

    fig = plt.figure()
    data.S.plot(**kwargs)
    ax = fig.gca()

    x0, y0 = ax.transAxes.transform((0, 0))  # lower left in pixels
    x1, y1 = ax.transAxes.transform((1, 1))  # upper right in pixes
    dx = x1 - x0
    dy = y1 - y0
    maxd = max(dx, dy)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    width = .03 * maxd / dx * (xlim[1] - xlim[0])
    height = .03 * maxd / dy * (ylim[1] - ylim[0])

    def onclick(event):
        ctx['points'].append([event.xdata, event.ydata])

        circ = matplotlib.patches.Ellipse((event.xdata, event.ydata,), width, height, color='red')
        ax.add_patch(circ)

        plt.draw()

    cid = plt.connect('button_press_event', onclick)

    return ctx['points']


