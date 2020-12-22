arpes.utilities.qt.windows module
=================================

**class arpes.utilities.qt.windows.SimpleWindow(\*args,**kwargs)\*\*

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
