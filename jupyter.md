Jupyter Lab Extensions
----------------------

If you are using Jupyter Lab as opposed to Jupyter, installation of the 
nbextensions: to do this install nodejs from nodejs.org and restart. 
Then open a terminal and run the labextension installation

Installing Python through Conda
-------------------------------

1. Install ``conda`` through the directions listed at
   `Anaconda <https://www.anaconda.com/download/>`__. You want the
   Python 3 version.
2. On UNIX like environments, create a Python 3.5 environment using the
   command
   ``conda create -n {name_of_your_env} python=3.5 scipy jupyter``. You
   can find more details in the conda docs at the `Conda User
   Guide <https://conda.io/docs/user-guide/tasks/manage-environments.html>`__.
   For Windows Users, launch Anaconda Navigator and create an
   environment.
3. In order to use your new environment, you can use the scripts
   ``source activate {name_of_your_env}`` and ``source deactivate``. Do
   this in any session where you will run your analysis or Jupyter. For
   Windows Users or users of graphical conda, launch your environment
   from the Navigator.