from typing import List

from arpes.typing import DataType
from arpes.utilities import normalize_to_spectrum

__all__ = ('decomposition_along', 'pca_along', 'ica_along',)


def decomposition_along(data: DataType, axes: List[str], decomposition_cls, correlation=False, **kwargs):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    flattened_data = normalize_to_spectrum(data).stack(fit_axis=axes)

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

    return into.unstack('fit_axis'), decomp


def pca_along(*args, **kwargs):
    from sklearn.decomposition import PCA
    return decomposition_along(*args, **kwargs, decomposition_cls=PCA)


def ica_along(*args, **kwargs):
    from sklearn.decomposition import FastICA
    return decomposition_along(*args, **kwargs, decomposition_cls=FastICA)