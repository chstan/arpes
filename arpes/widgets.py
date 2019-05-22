import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from matplotlib.widgets import LassoSelector, Button, TextBox, RectangleSelector
from matplotlib.path import Path

from arpes.plotting.utils import imshow_arr


__all__ = ('pick_rectangles', 'pick_points', 'pca_explorer',)


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


@popout
def pca_explorer(pca, data, component_dim='components', initial_values=None, **kwargs):
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

    gs = gridspec.GridSpec(2, 2)
    ax_components = plt.subplot(gs[0, 0])
    ax_sum_selected = plt.subplot(gs[0, 1])
    ax_map = plt.subplot(gs[1, 0])

    gs_widget = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1])
    ax_widget_1 = plt.subplot(gs_widget[0, 0])
    ax_widget_2 = plt.subplot(gs_widget[1, 0])
    ax_widget_3 = plt.subplot(gs_widget[2, 0])

    _, data_imshow = imshow_arr(data.sum(pca_dims), ax=ax_sum_selected, cmap='viridis')
    _, map_imshow = imshow_arr(data.sum(other_dims), ax=ax_map)

    def update_from_selection(ind):
        # calculate the new data
        if ind is None or len(ind) == 0:
            stacked = data.stack(pca_dims=pca_dims).sum('pca_dims')
            context['selected_indices'] = []
        else:
            context['selected_indices'] = ind
            stacked = data.stack(pca_dims=pca_dims).isel(pca_dims=ind).sum('pca_dims')

        context['sum_data'] = stacked

        ax_map.clear()

        if context['integration_region'] is not None:
            data_sel = data.sel(**context['integration_region']).sum(other_dims)
        else:
            data_sel = data.sum(other_dims)

        _, imshowed = imshow_arr(data_sel, ax=ax_map)

        s = data_sel.values.shape
        data_sel = (data_sel.values * 0 + 1)
        mask = (data_sel == 0).ravel()
        mask[ind] = True
        mask = mask.reshape(s)

        my_cmap = matplotlib.cm.Reds
        my_cmap.set_bad('k', alpha=0)
        data_sel = np.ma.masked_where(np.logical_not(mask), data_sel)
        ax_map.imshow(data_sel.T, cmap=my_cmap, interpolation='none', alpha=0.35, vmax=1, vmin=0,
                      origin='lower', extent=imshowed.get_extent(), aspect=ax_map.get_aspect())

        # update the sum display
        data_imshow.set_data(stacked.values)
        data_imshow.autoscale()

    def set_axes(component_x, component_y):
        ax_components.clear()
        context['selected_components'] = [component_x, component_y]
        for_scatter, size = compute_for_scatter()
        pts = ax_components.scatter(for_scatter.values[0], for_scatter.values[1], s=size)
        if context['selector'] != None:
            context['selector'].disconnect()

        context['selector'] = SelectFromCollection(ax_components, pts, on_select=update_from_selection)
        update_from_selection([])

    def on_select_summed(event_click, event_release):
        x1, y1 = event_click.xdata, event_click.ydata
        x2, y2 = event_release.xdata, event_release.ydata

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        context['integration_region'] = dict([[other_dims[1], slice(x1, x2)], [other_dims[0], slice(y1, y2)]])
        update_from_selection(context['selected_indices'])
        context['rect_selector'].update()

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
    context['axis_X_input'] = TextBox(ax_widget_2, 'Axis X:', initial=str(initial_values[0]))
    context['axis_Y_input'] = TextBox(ax_widget_3, 'Axis Y:', initial=str(initial_values[1]))
    context['axis_button'].on_clicked(on_change_axes)

    context['rect_selector'] = RectangleSelector(
        ax_sum_selected, on_select_summed, drawtype='box',
        rectprops=dict(fill=False, edgecolor='black', linewidth=2), lineprops=dict(linewidth=2, color='black')
    )

    set_axes(*initial_values)

    plt.tight_layout()
    return context


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


