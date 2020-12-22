arpes.utilities.qt package
==========================

Submodules
----------

-   [arpes.utilities.qt.app module](arpes.utilities.qt.app)
-   [arpes.utilities.qt.data\_array\_image\_view
    module](arpes.utilities.qt.data_array_image_view)
-   [arpes.utilities.qt.help\_dialogs
    module](arpes.utilities.qt.help_dialogs)
-   [arpes.utilities.qt.windows module](arpes.utilities.qt.windows)

Module contents
---------------

**class arpes.utilities.qt.BasicHelpDialog(shortcuts=None)**

> Bases: `PyQt5.QtWidgets.QDialog`
>
> **keyPressEvent(self, QKeyEvent)**

**class arpes.utilities.qt.DataArrayImageView(root, \*args,**kwargs)\*\*

> Bases: `pyqtgraph.imageview.ImageView.ImageView`
>
> ImageView that transparently handles xarray data, including setting
> axis and coordinate information.
>
> This makes it easier to build interactive applications around
> realistic scientific datasets.
>
> **recompute()**
>
> **setImage(img, keep\_levels=False, \*args,**kwargs)\*\*
>
> > Accepts an xarray.DataArray instead of a numpy array :param img:
> > :param args: :param kwargs: :return:

**class arpes.utilities.qt.SimpleApp**

> Bases: `object`
>
> Has all of the layout information and business logic for an
> interactive data browsing utility using PyQt5.
>
> `DEFAULT_COLORMAP = 'viridis'`
>
> `TITLE = 'Untitled Tool'`
>
> `WINDOW_CLS = None`
>
> `WINDOW_SIZE = (4, 4)`
>
> **after\_show()**
>
> **before\_show()**
>
> **static build\_pg\_cmap(colormap)**
>
> **generate\_marginal\_for(dimensions, column, row, name=None,
> orientation='horiz', cursors=False, layout=None)**
>
> **layout()**
>
> **property ninety\_eight\_percentile**
>
> **print(\*args,**kwargs)\*\*
>
> **set\_colormap(colormap)**
>
> **start()**

**class arpes.utilities.qt.SimpleWindow(\*args,**kwargs)\*\*

> Bases: `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtCore.QObject`
>
> Provides a relatively simple way of making a windowed application with
> the following utilities largely managed for you:
>
> 1.  Keybindings and chords
> 2.  Help and info
> 3.  Lifecycle
> 4.  Inter-component messaging
>
> `HELP_DIALOG_CLS = None`
>
> **close(self) -&gt; bool**
>
> **compile\_cursor\_modes()**
>
> **compile\_key\_bindings()**
>
> **do\_close(event)**
>
> **eventFilter(self, QObject, QEvent) -&gt; bool**
>
> **handleKeyPressEvent(event)**
>
> **toggle\_help(event)**
>
> **window\_print(\*args,**kwargs)\*\*
