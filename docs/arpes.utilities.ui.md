arpes.utilities.ui module
=========================

Easily composable and reactive UI utilities using RxPy and PyQt5. This
makes UI prototyping *MUCH* faster. In order to log IDs so that you can
attach subscriptions after the fact, you will need to use the CollectUI
context manager.

An example is as follows, showing the currently available widgets. If
you don’t need to attach callbacks, you can get away without using the
context manager.

&gt;&gt;`` <<` ui = {} with CollectUI(ui):  ..     test_widget = grid(       group(          text_edit(‘starting text’, id=’text’), line_edit(‘starting          line’, id=’line’), combo_box([‘A’, ‘B’, ‘C’], id=’combo’),          spin_box(5, id=’spinbox’), radio_button(‘A Radio’,          id=’radio’), check_box(‘Checkbox’, id=’check’),          slider(id=’slider’), file_dialog(id=’file’), button(‘Send          Text’, id=’submit’)        ), widget=self,     )  >> ``&lt;&lt;&gt;&gt;\`&lt;&lt;

“Forms” can effectively be built by building an observable out of the
subjects in the UI. We have a *submit* function that makes creating such
an observable simple.

`` ` submit('submit', ['check', 'slider', 'file'], ui).subscribe(lambda item: print(item)) ``\`

With the line above, whenever the button with id=’submit’ is pressed, we
will log a dictionary with the most recent values of the inputs
{‘check’,’slider’,’file’} as a dictionary with these keys. This allows
building PyQt5 “forms” without effort.

**class arpes.utilities.ui.CollectUI(target\_ui=None)**

> Bases: `object`

**class arpes.utilities.ui.CursorRegion(\*args,**kwargs)\*\*

> Bases: `pyqtgraph.graphicsItems.LinearRegionItem.LinearRegionItem`
>
> **lineMoved(i)**
>
> **set\_location(value)**
>
> **set\_width(value)**

**class arpes.utilities.ui.KeyBinding(label, chord, handler)**

> Bases: `tuple`
>
> `chord`
>
> > Alias for field number 1
>
> `handler`
>
> > Alias for field number 2
>
> `label`
>
> > Alias for field number 0

**arpes.utilities.ui.bind\_dataclass(dataclass\_instance, prefix: str,
ui: Dict\[str, PyQt5.QtWidgets.QWidget\])**

> One-way data binding between a dataclass instance and a collection of
> widgets in the UI.
>
> Sets the current UI state to the value of the Python dataclass
> instance, and sets up subscriptions to value changes on the UI so that
> any future changes are propagated to the dataclass instance.
>
> Parameters  
> -   **dataclass\_instance** – Instance to link
> -   **prefix** – Prefix for widget IDs in the UI
> -   **ui** – Collected UI elements
>
> Returns  

**arpes.utilities.ui.button(text, \*args)**

**arpes.utilities.ui.check\_box(text, \*args)**

**arpes.utilities.ui.combo\_box(items, \*args, name=None)**

**arpes.utilities.ui.file\_dialog(\*args)**

**arpes.utilities.ui.grid(\*children, layout\_cls=&lt;class
'PyQt5.QtWidgets.QGridLayout'&gt;, widget=None)**

**arpes.utilities.ui.group(\*args, label=None, layout\_cls=None)**

**arpes.utilities.ui.horizontal(\*children, layout\_cls=&lt;class
'PyQt5.QtWidgets.QHBoxLayout'&gt;, widget=None)**

**arpes.utilities.ui.label(text, \*args,**kwargs)\*\*

**arpes.utilities.ui.layout(\*children, layout\_cls=None, widget=None)**

**arpes.utilities.ui.layout\_dataclass(dataclass\_cls, prefix:
Optional\[str\] = None)**

> Renders a dataclass instance to QtWidgets. See also *bind\_dataclass*
> below to get one way data binding to the instance :param
> dataclass\_cls: :param prefix: :return:

**arpes.utilities.ui.line\_edit(\*args)**

**arpes.utilities.ui.numeric\_input(value=0, input\_type: type =
&lt;class 'float'&gt;, \*args, validator\_settings=None)**

**arpes.utilities.ui.pretty\_key\_event(event)**

> Key Event -&gt; List\[str\] in order to be able to prettily print keys
> :param event: :return:

**arpes.utilities.ui.radio\_button(text, \*args)**

**arpes.utilities.ui.slider(minimum=0, maximum=10, interval=None,
horizontal=True)**

**arpes.utilities.ui.spin\_box(minimum=0, maximum=10, step=1,
adaptive=True, value=None)**

**arpes.utilities.ui.splitter(first, second, direction=2, size=None)**

**arpes.utilities.ui.submit(gate: str, keys: List\[str\], ui: Dict\[str,
PyQt5.QtWidgets.QWidget\]) -&gt;
rx.core.observable.observable.Observable**

**arpes.utilities.ui.tabs(\*children)**

**arpes.utilities.ui.text\_edit(text='', \*args)**

**arpes.utilities.ui.vertical(\*children, layout\_cls=&lt;class
'PyQt5.QtWidgets.QVBoxLayout'&gt;, widget=None)**
