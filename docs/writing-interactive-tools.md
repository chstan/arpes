# Tutorial: Adding an Interactive Tool

This section provides a guided tutorial on adding an interactive utility to PyARPES. 
Internally, support is provided to make it straightforward to add an interactivity using

1. **bokeh**: an option for interactivity inside Jupyter
2. **matplotlib + Qt**: interactive popout Matplotlib figures
3. **Native Qt Applications**: Fast native applications for heavier applications

This section will cover the third option and show how Qt5 and pyqtgraph can be used to make an interactive tool
that allows you to tune function parameters applied to a spectrum. Examples of the other options can be found in
the source for `bokeh_tool/.S.show` (`bokeh`) and `widgets.py` (`matplotlib`).

## Structure

We will put our utility into two files, one which will contain the code for the tool, 
another with a manual test harness so we can start it quickly. Relative to the source root, these files
can be

1. `arpes/plotting/dynamic_tool.py`
2. `scripts/test_dynamic_tool.py`

## Test Harness

Let's first write the test script. This is straightforward since we can
use the example data with our function using `arpes.io.load_example_data`.

In order to provide a parameterized function that we can test on the data we
we can also define the function `adjust_gamma` which rescales the intensity in a spectrum
according to $x \rightarrow x^\gamma$.

Finally, we can invoke our not-yet existent function `make_dynamic`
which we can decide will take the function as the first argument and the data as
the second argument.

```python
from arpes.io import load_example_data
from arpes.plotting.dynamic_tool import make_dynamic

data = load_example_data()


def adjust_gamma(data, gamma: float = 1):
    """
    Equivalent to adjusting the display gamma factor, just rescale the data
    according to f(x) = x^gamma.

    :param data:
    :param gamma:
    :return:
    """
    return data ** gamma


make_dynamic(adjust_gamma, data)
```

To test our code, we can now run this script which will handle the 
imports and data loading for us. Currently, it fails because we have 
not yet defined the module `arpes.plotting.dynamic_tool`

## Writing the Interactive Utility

Now that we have a script we can use to call our utility while we write it, we can focus
on scripting the UI and dealing with the business logic.

### Application Structure and Lifecycle

Qt-based applications in PyARPES are managed according to their lifecycle. A summary of the lifecycle of an interactive 
utility in PyARPES is included here for reference. 

<img style="margin-left: calc(50% - 200px)" width="400px" src="/static/Lifecycle-Diagram.svg" data-origin="static/Lifecycle-Diagram.svg" alt />

For the most part, you should not need to modify the parts in blue boxes, although you can do so if needed.
Generally, you will need to fill out the `layout` function as well as `before_show` where you may perform 
additional layout logic or initialization as required by your application.

Although not typically required a hook `after_show` is also available in case any Qt or pyqtgraph settings need
adjusting after the initial draw to the screen.

Once running, interaction is provided through mouse and keyboard handlers, until a request is made to close it. This can occur 
due to an unhandled `Exception`, the user pressing `Esc`, the user clicking the window close button, or programmatically. 


### Getting a Window 

There's a bit of starter code we need before we get too deep. As a first step
we will add only as much as is needed to spawn a window.

```python
from PyQt5 import QtWidgets
from arpes.utilities.qt import qt_info, SimpleApp, SimpleWindow, BasicHelpDialog


__all__ = ('DynamicTool', 'make_dynamic',)

qt_info.setup_pyqtgraph()


class DynamicToolWindow(SimpleWindow):
    HELP_DIALOG_CLS = BasicHelpDialog


class DynamicTool(SimpleApp):
    WINDOW_SIZE = (5,5,) # 5 inches by 5 inches
    WINDOW_CLS = DynamicToolWindow
    TITLE = '' # we will use the function name for the window title

    def __init__(self, function):
        self._function = function
        self.main_layout = QtWidgets.QGridLayout()
        self.data = None

        super().__init__()

    def set_data(self, data):
        pass

    def layout(self):
        return self.main_layout


def make_dynamic(fn, data):
    tool = DynamicTool(fn)
    tool.set_data(data)
    tool.start()
```

As we can see, we added two classes `DynamicToolWindow`, which handles responding to events 
like keyboard shortcuts, and `DynamicTool`, which handles the logic of our application and is 
responsible for actually drawing and delegating the UI.

We will add more code to each later. For now, notice that we declare a help page class via
`HELP_DIALOG_CLS = BasicHelpDialog`: this will provide a help panel summarizing the available keyboard
shortcuts. You can open this panel by pressing 'H' on any running interactive application in PyARPES.

In `DynamicTool`, we set the window size desired in inches on the screen, as well as the window 
class which we set to `DynamicToolWindow`. We can leave the window title blank because later we will set it 
to the name of the function we call the panel with.

### Displaying Data

A next step is to display the original data.

```python
class DynamicTool(SimpleApp):
    ...

    def configure_image_widgets(self):
        self.generate_marginal_for((), 0, 0, 'xy', cursors=False, layout=self.content_layout)
        self.generate_marginal_for((), 1, 0, 'f(xy)', cursors=False, layout=self.content_layout)
        self.main_layout.addLayout(self.content_layout, 0, 0)

    def update_data(self):
        self.views['xy'].setImage(self.data.S.nan_to_num())
        self.views['f(xy)'].setImage(self.data.S.nan_to_num()) # for now just display the same data

    def add_controls(self):
        pass

    def before_show(self):
        self.configure_image_widgets()
        self.add_controls()
        self.update_data()

    def set_data(self, data: DataType):
        self.data = normalize_to_spectrum(data)
```

To display the data, we added logic in the `before_show` [lifecycle hook](#application-structure-and-lifecycle) to generate
plots for the data (`xy`) and the transformed data (`f(xy)`). To do this we used the utility function `generate_marginal_for` 
that can be used to create browsable marginal plots for high dimensional data sets. Here we do not want to integrate out 
any dimensions so we passed an tuple as the first argument. With the rest of the invokation we specify to add the plot
to the layout `self.content_layout` in the locations (0,0) and (1,0). Because we are not linking plots we don't need cursors.

We can also add a function `update_data` in order to set the data on each of these views.

### Generating controls from the function

In order to generate controls, we need to know the types and default values of arguments. Using `inspect.getfullargspec`
we can introspect the arguments to the supplied function. We will make the reasonable assumption that the user is supplying
type annotations but that default values may not be available.

```python
class DynamicTool(SimpleApp):
    ...
    def __init__(self, function):
        ...
        self.current_arguments = {}

        super().__init__()

    def calculate_control_specification(self):
        argspec = inspect.getfullargspec(self._function)

        # we assume that the first argument is the input data
        args = argspec.args[1:]

        defaults_for_type = {
            float: 0.,
            int: 0,
            str: '',
        }

        specs = []
        for i, arg in enumerate(args[::-1]):
            argument_type = argspec.annotations.get(arg, float)
            if i < len(argspec.defaults):
                argument_default = argspec.defaults[len(argspec.defaults) - (i+1)]
            else:
                argument_default = defaults_for_type.get(argument_type, 0)

            self.current_arguments[arg] = argument_default
            specs.append([
                arg,
                argument_type,
                argument_default,
            ])

        return specs
```

Now that we can generate from a type annotated function a description of the parameters, we
can use this to generate UI inputs (controls) for these parameters and render them into our utility.

For each control, we will "subscribe" to changes in the value so that we can update the plot
with the new value of the function called with the updated parameters. To do this, we will add two new functions
`add_controls` and `build_control_for`. `build_control_for` is simple, it just takes the description of the parameter
we computed with `calculate_control_specification` and returns an appropriate widget. Notice that we pass an ID with the
parameter name when we construct the UI element. This allows us to find the control later and subscribe to changes.

In `add_controls` we:

1. Calculate the parameter description
2. Inside `CollectUI(ui)` iterate across these parameter descriptions and group them into a 
   "Controls" tab of the UI.
3. Iterate across the controls and attach a function which responds to changes in the UI `update_argument`
4. Add the tabbed region to the main UI (`self.main_layout.addWidget(controls, 1, 0)`)

Finally, in `update_data` we modify the code to invoke the function with the new parameters
and update the view.
 

```python
class DynamicTool(SimpleApp):
    ...
    def update_data(self):
        self.views['xy'].setImage(self.data.S.nan_to_num())
        try:
            mapped_data = self._function(self.data, **self.current_arguments)
            self.views['f(xy)'].setImage(mapped_data.S.nan_to_num())
        except:
            pass

    def add_controls(self):
        specification = self.calculate_control_specification()

        ui = {}
        with CollectUI(ui):
            controls = tabs(
                ['Controls', horizontal(
                    *[vertical(*[vertical(label(s[0]), self.build_control_for(*s)) for s in pair])
                      for pair in group_by(2, specification)])],
            )

        def update_argument(arg_name, arg_type):
            def updater(value):
                self.current_arguments[arg_name] = arg_type(value)
                self.update_data()

            return updater

        for arg_name, arg_type, _ in specification:
            ui[f'{arg_name}-control'].subject.subscribe(update_argument(arg_name, arg_type))

        controls.setFixedHeight(qt_info.inches_to_px(1.4))
        self.main_layout.addWidget(controls, 1, 0)

    def build_control_for(self, parameter_name, parameter_type, parameter_default):
        if parameter_type in (int, float,):
            return numeric_input(parameter_default, parameter_type, id=f'{parameter_name}-control')

        if parameter_type == str:
            return line_edit(parameter_default, id=f'{parameter_name}-control')
```

All together, this is about 100 lines of code to make a native, interactive application that allows you to make any 
analysis function interactive. Not too bad!