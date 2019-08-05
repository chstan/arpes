from typing import List

from functools import wraps

from arpes.provenance import provenance
from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum

__all__ = ('decomposition_along', 'pca_along', 'ica_along', 'factor_analysis_along',)


def decomposition_along(data: DataType, axes: List[str], decomposition_cls, correlation=False, **kwargs):
    """
    Performs a change of basis of your data according to `sklearn` decomposition classes. This allows
    for robust and simple PCA, ICA, factor analysis, and other decompositions of your data even when it
    is very high dimensional.

    Generally speaking, PCA and similar techniques work when data is 2D, i.e. a sequence of 1D observations.
    We can make the same techniques work by unravelling a ND dataset into 1D (i.e. np.ndarray.ravel()) and
    unravelling a KD set of observations into a 1D set of observations. This is basically grouping axes. As
    an example, if you had a 4D dataset which consisted of 2D-scanning valence band ARPES, then the dimensions
    on our dataset would be "[x,y,eV,phi]". We can group these into [spatial=(x, y), spectral=(eV, phi)] and
    perform PCA or another analysis of the spectral features over different spatial observations.

    If our data was called `f`, this can be accomplished with:

    ```
    transformed, decomp = decomposition_analysis(f.stack(spectral=['eV', 'phi']), ['x', 'y'], PCA)
    transformed.dims # -> [X, Y, components]
    ```

    The results of `decomposition_along` can be explored with `arpes.widgets.pca_explorer`, regardless of
    the decomposition class.

    :param data: Input data, can be N-dimensional but should only include one "spectral" axis.
    :param axes: Several axes to be treated as a single axis labeling the list of observations.
    :param decomposition_cls: A sklearn.decomposition class (such as PCA or ICA) to be used
                              to perform the decomposition.
    :param correlation: Controls whether StandardScaler() is used as the first stage of the data ingestion
                        pipeline for sklearn.
    :param kwargs:
    :return:
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler


    if len(axes) > 1:
        flattened_data = normalize_to_spectrum(data).stack(fit_axis=axes)
        stacked = True
    else:
        flattened_data = normalize_to_spectrum(data).S.transpose_to_back(axes[0])
        stacked = False

    if len(flattened_data.dims) != 2:
        raise ValueError('Inappropriate number of dimensions after flattening: [{}]'.format(
            flattened_data.dims))

    if correlation:
        pipeline = make_pipeline(StandardScaler(), decomposition_cls(**kwargs))
    else:
        pipeline = make_pipeline(decomposition_cls(**kwargs))

    pipeline.fit(flattened_data.values.T)

    decomp = pipeline.steps[-1][1]

    transform = decomp.transform(flattened_data.values.T)

    into = flattened_data.copy(deep=True)
    into_first = into.dims[0]
    into = into.isel(**dict([[into_first, slice(0, transform.shape[1])]]))
    into = into.rename(dict([[into_first, 'components']]))

    into.values = transform.T

    if stacked:
        into = into.unstack('fit_axis')

    provenance(into, data, {
        'what': 'sklearn decomposition',
        'by': 'decomposition_along',
        'axes': axes,
        'correlation': False,
        'decomposition_cls': decomposition_cls.__name__,
    })

    return into, decomp


@wraps(decomposition_along)
def pca_along(*args, **kwargs):
    from sklearn.decomposition import PCA
    return decomposition_along(*args, **kwargs, decomposition_cls=PCA)


@wraps(decomposition_along)
def factor_analysis_along(*args, **kwargs):
    from sklearn.decomposition import FactorAnalysis
    return decomposition_along(*args, **kwargs, decomposition_cls=FactorAnalysis)


@wraps(decomposition_along)
def ica_along(*args, **kwargs):
    from sklearn.decomposition import FastICA
    return decomposition_along(*args, **kwargs, decomposition_cls=FastICA)