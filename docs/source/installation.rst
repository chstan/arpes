.. _installation:

Installation
============

Some common issues in installation have been written up in the
`FAQ </faq>`__.

You can install PyARPES in an editable configuration so that you can
edit it to your needs (recommended) or as a standalone package from a
package manager. In the latter case, you should put any custom code in a
separate module which you import together with PyARPES to serve your
particular analysis needs.

Installation from Source
~~~~~~~~~~~~~~~~~~~~~~~~

Using an installation from source is the best option if you want to
frequently change the source of PyARPES as you work. You can use code
available either from the main repository at
`GitLab <https://gitlab.com/lanzara-group/python-arpes.git>`__ or the
`GitHub mirror <https://github.com/chstan/arpes>`__.

1. **Install Miniconda or Anaconda** according to the
   `directions <https://docs.conda.io/en/latest/miniconda.html>`__
2. Clone or otherwise download the respository

.. code:: bash

   git clone https://gitlab.com/lanzara-group/python-arpes

3. Make a conda environment according to our provided specification

.. code:: bash

   cd path/to/python-arpes
   conda env create -f environment.yml

3. Activate the environment

.. code:: bash

   conda activate arpes

4. Install PyARPES in an editable configuration

.. code:: bash

   pip install -e .

5. *Recommended:* Configure IPython kernel according to the **Barebones
   Kernel Installation** below

From Package Managers
~~~~~~~~~~~~~~~~~~~~~

It is highly recommended that you install PyARPES through ``conda``
rather than ``pip``. You will also need to specify ``conda-forge`` as a
channel in order to pick up a few dependencies. Make sure you don’t add
conda-forge with higher priority than the Anaconda channel, as this
might cause issues with installing BLAS into your environment. We
recommend

.. code:: bash

   conda config --append channels conda-forge
   conda install -c arpes arpes

Additional Suggested Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install and configure standard tools like
   `Jupyter <https://jupyter.org/>`__ or Jupyter Lab. Notes on
   installing and configuring Jupyter based installations can be found
   in ``jupyter.md``
2. Explore the documentation and example notebooks at `the documentation
   site <https://arpes.netlify.com/>`__.

Barebones kernel installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have Jupyter and just need to register your environment.
You can do

.. code:: bash

   pip install ipykernel
   python -m ipykernel install --user 

You can also give the kernel a different display name in Juptyer with
``python -m ipykernel install --user --display-name "My Name Here"``.

