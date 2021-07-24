Workspaces
----------

*Note for old users*: Workspaces are friendlier than before. You
shouldn’t require any real setup to use them now. If something
resembling a workspace can’t be found, PyARPES now assumes that your
current working directory is a workspace.

**Workspaces** are a concept PyARPES uses to organize data, figures, and
analysis logs. For most uses, you won’t need to specify workspaces, but
the high level ideas are as follows:

1. Put data, notebooks, figures pertaining to different physics projects
   in different folders. The resulting bundle/directory is called a
   **workspace**.
2. PyARPES will organize figures you make according to the date and
   workspace you were in, if you use ``arpes.plotting.utils.savefig``
   instead of ``matplotlib.pyplot.savefig``.
3. PyARPES will attempt to keep full analysis logs for you, for each
   session.

PyARPES lets you programmatically control the workspace if you need to,
for instance to conveniently load data or results from a different
project.
