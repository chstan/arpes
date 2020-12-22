arpes.workflow module
=====================

-   go\_to\_figures
-   go\_to\_workspace
-   go\_to\_cwd
-   publish\_data
-   read\_data
-   consume\_data
-   summarize\_data

This module contains some extra utilities for managing scientific
workflows, especially between notebooks and workspaces.

A core feature of this module is that you can export and import data
from between notebooks. *easy\_pickle* also fulfills this to an extent,
but with the tools included here, there are some extra useful goodies,
like tracking of publishing and consuming notebooks, so that you get
reminders about where your data comes from if you need to regenerate it.

This isn’t dataflow for Jupyter notebooks, but it is at least more
convenient than managing it all yourself.

**arpes.workflow.consume\_data(key='\*', workspace=None)**

**arpes.workflow.go\_to\_cwd()**

> Opens the current working directory. :return:

**arpes.workflow.go\_to\_figures()**

> Opens the figures folder for the current workspace and the current
> day, otherwise finds the most recent one and opens it. :return:

**arpes.workflow.go\_to\_workspace(workspace=None)**

> Opens the workspace folder, otherwise opens the location of the
> running notebook. :return:

**arpes.workflow.publish\_data(key, data, workspace)**

**arpes.workflow.read\_data(key='\*', workspace=None)**

**arpes.workflow.summarize\_data(key=None, workspace=None)**
