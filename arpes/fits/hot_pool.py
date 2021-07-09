"""A lazy keep-alive `multiprocesssing.Pool`.

We keep a pool alive after one is requested at the cost of memory overhead
because otherwise pools are too slow due to heavy analysis imports (scipy, etc.).
"""

from multiprocessing import Pool, pool

# from pathos.pools import ProcessPool
from typing import Optional

__all__ = ["hot_pool"]


class HotPool:
    _pool: Optional[pool.Pool] = None

    @property
    def pool(self) -> pool.Pool:
        if self._pool is not None:
            return self._pool

        self._pool = Pool()
        return self._pool

    def __del__(self):
        if self._pool is not None:
            self._pool.close()
            self._pool = None


hot_pool = HotPool()
