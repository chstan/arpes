import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import numpy as np
from arpes.provenance import save_plot_provenance
from arpes.analysis import filters
from mpl_toolkits.mplot3d import Axes3D # need this import to enable 3D axes

from .utils import *

__all__ = ['plot_isosurface', 'plot_trisurf', 'plotly_trisurf']


@save_plot_provenance
def plot_isosurface(data, level=None, percentile=99.5, smoothing=None, out=None, ax=None):
    assert ('eV' in data.dims)

    new_dim_order = list(data.dims)
    new_dim_order.remove('eV')
    new_dim_order = new_dim_order + ['eV']
    data = data.transpose(*new_dim_order)
    colormap = plt.get_cmap('viridis')

    if smoothing is not None:
        data = filters.gaussian_filter_arr(data, sigma=smoothing)

    spacing = [data.coords[d][1] - data.coords[d][0] for d in data.dims]

    if level is None:
        level = np.percentile(data.data, percentile)

    from skimage import measure # try to avoid skimage incompatibility with numpy v0.16 as much as possible
    verts, faces, normals, values = measure.marching_cubes(data.values, level, spacing=spacing)

    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                    lw=1, facecolors=colormap(values))

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()


def plotly_trisurf(data):
    import plotly.plotly as py
    import plotly.offline as pyo
    import plotly.graph_objs as go

    x = data.coords[data.dims[0]].values
    y = data.coords[data.dims[1]].values
    plot_data = [
        go.Surface(
            x=x,
            y=y,
            z=data.transpose().values,
        )
    ]
    layout = go.Layout(
        title='Test Title',
        autosize=True,
        width=900,
        height=900,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )
    fig = go.Figure(data=plot_data, layout=layout)
    pyo.iplot(fig, filename='test-surface')


def plot_trisurf(data, out=None, ax=None, angles=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    Xs, Ys = np.meshgrid(data.coords[data.dims[0]].values, data.coords[data.dims[1]].values)
    triang = Triangulation(Xs.ravel(), Ys.ravel())
    mask = np.any(np.isnan(data.transpose().values.ravel()[triang.triangles]), axis=1)
    triang.set_mask(mask)

    values = data.copy(deep=True).transpose().values
    values[np.isnan(values)] = 0
    ax.plot_trisurf(triang, values.ravel(), antialiased=True, cmap=cm.coolwarm, linewidth=0)

    if angles is not None:
        ax.view_init(*angles)

    #plt.show()
    return ax

@save_plot_provenance
def plot_trisurface(data, out=None, ax=None):
    if ax is None:
        pass

    # -----------------------------------------------------------------------------
    # Generating the initial data test points and triangulation for the demo
    # -----------------------------------------------------------------------------
    # User parameters for data test points
    n_test = 200  # Number of test data points, tested from 3 to 5000 for subdiv=3

    subdiv = 3  # Number of recursive subdivisions of the initial mesh for smooth
    # plots. Values >3 might result in a very high number of triangles
    # for the refine mesh: new triangles numbering = (4**subdiv)*ntri

    init_mask_frac = 0.0  # Float > 0. adjusting the proportion of
    # (invalid) initial triangles which will be masked
    # out. Enter 0 for no mask.

    min_circle_ratio = .01  # Minimum circle ratio - border triangles with circle
    # ratio below this will be masked if they touch a
    # border. Suggested value 0.01 ; Use -1 to keep
    # all triangles.

    # Random points
    random_gen = np.random.mtrand.RandomState(seed=127260)
    x_test = random_gen.uniform(-1., 1., size=n_test)
    y_test = random_gen.uniform(-1., 1., size=n_test)
    z_test = experiment_res(x_test, y_test)

    # meshing with Delaunay triangulation
    x_dim, y_dim = tuple(data.dims)
    x_coord, y_coord = data.coords[x_dim].data, data.coords[y_dim].data

    tri = Triangulation(data.coords[data.dims[0]].data, y_test)
    ntri = tri.triangles.shape[0]

    # Some invalid data are masked out
    mask_init = np.zeros(ntri, dtype=np.bool)
    masked_tri = random_gen.randint(0, ntri, int(ntri * init_mask_frac))
    mask_init[masked_tri] = True
    tri.set_mask(mask_init)

    # -----------------------------------------------------------------------------
    # Improving the triangulation before high-res plots: removing flat triangles
    # -----------------------------------------------------------------------------
    # masking badly shaped triangles at the border of the triangular mesh.
    mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)
    tri.set_mask(mask)

    # refining the data
    refiner = UniformTriRefiner(tri)
    tri_refi, z_test_refi = refiner.refine_field(z_test, subdiv=subdiv)

    # analytical 'results' for comparison
    z_expected = experiment_res(tri_refi.x, tri_refi.y)

    # for the demo: loading the 'flat' triangles for plot
    flat_tri = Triangulation(x_test, y_test)
    flat_tri.set_mask(~mask)

    # -----------------------------------------------------------------------------
    # Now the plots
    # -----------------------------------------------------------------------------
    # User options for plots
    plot_tri = True  # plot of base triangulation
    plot_masked_tri = True  # plot of excessively flat excluded triangles
    plot_refi_tri = False  # plot of refined triangulation
    plot_expected = False  # plot of analytical function values for comparison

    # Graphical options for tricontouring
    levels = np.arange(0., 1., 0.025)

    plt.figure()
    plt.gca().set_aspect('equal')
    plt.title("Filtering a Delaunay mesh\n" +
              "(application to high-resolution tricontouring)")

    # 1) plot of the refined (computed) data contours:
    plt.tricontour(tri_refi, z_test_refi, levels=levels, cmap='Blues',
                   linewidths=[2.0, 0.5, 1.0, 0.5])
    # 2) plot of the expected (analytical) data contours (dashed):
    if plot_expected:
        plt.tricontour(tri_refi, z_expected, levels=levels, cmap='Blues',
                       linestyles='--')
    # 3) plot of the fine mesh on which interpolation was done:
    if plot_refi_tri:
        plt.triplot(tri_refi, color='0.97')
    # 4) plot of the initial 'coarse' mesh:
    if plot_tri:
        plt.triplot(tri, color='0.7')
    # 4) plot of the unvalidated triangles from naive Delaunay Triangulation:
    if plot_masked_tri:
        plt.triplot(flat_tri, color='red')

    plt.show()

    if out is not None:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()

