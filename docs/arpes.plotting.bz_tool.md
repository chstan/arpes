# arpes.plotting.bz\_tool package

## Submodules

  - [arpes.plotting.bz\_tool.CoordinateOffsetWidget
    module](arpes.plotting.bz_tool.CoordinateOffsetWidget)

## Module contents

**class arpes.plotting.bz\_tool.BZTool**

> Bases: `object`
> 
> Implements a Brillouin zone explorer showing the region of momentum
> probed by ARPES.
> 
> **add\_coordinate\_control\_widgets()**
> 
> **configure\_main\_widget()**
> 
> **construct\_coordinate\_info\_tab()**
> 
> **construct\_detector\_info\_tab()**
> 
> **construct\_general\_settings\_tab()**
> 
> **construct\_sample\_info\_tab()**
> 
> `coordinates`
> 
> **on\_change\_material(value)**
> 
> **start()**
> 
> **update\_cut(\*args)**

**class arpes.plotting.bz\_tool.BZToolWindow(\*args,**kwargs)\*\*

> Bases: `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtCore.QObject`
> 
> **close(self) -\> bool**
> 
> **do\_close(event)**
> 
> **eventFilter(self, QObject, QEvent) -\> bool**
> 
> **handleKeyPressEvent(event)**
> 
> **window\_print(\*args,**kwargs)\*\*

**arpes.plotting.bz\_tool.bz\_tool()**
