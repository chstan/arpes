# arpes.utilities.dataset module

**arpes.utilities.dataset.clean\_xlsx\_dataset(path,
allow\_soft\_match=False, write=True, with\_inferred\_cols=True,
warn\_on\_exists=False,**kwargs)\*\*

**arpes.utilities.dataset.default\_dataset(workspace=None,
match=None,**kwargs)\*\*

**arpes.utilities.dataset.infer\_data\_path(file, scan\_desc,
allow\_soft\_match=False,
use\_regex=True)**

**arpes.utilities.dataset.attach\_extra\_dataset\_columns(path,**kwargs)\*\*

**arpes.utilities.dataset.swap\_reference\_map(df:
pandas.core.frame.DataFrame, old\_reference, new\_reference)**

> Replaces instances of a reference map old\_reference in the ref\_map
> column with new\_reference. :param df: :return:

**arpes.utilities.dataset.cleaned\_dataset\_exists(path)**

**arpes.utilities.dataset.modern\_clean\_xlsx\_dataset(path,
allow\_soft\_match=False, with\_inferred\_cols=True,
write=False,**kwargs)\*\*

**arpes.utilities.dataset.cleaned\_pair\_paths(path)**

**arpes.utilities.dataset.list\_files\_for\_rename(path=None,
extensions=None)**

**arpes.utilities.dataset.rename\_files(dry=True, path=None,
extensions=None, starting\_index=1)**
