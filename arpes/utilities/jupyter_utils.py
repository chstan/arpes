from tqdm import tqdm_notebook

__all__ = ('wrap_tqdm',)


def wrap_tqdm(x, interactive=True, *args, **kwargs):
    if not interactive:
        return x

    return tqdm_notebook(x, *args, **kwargs)