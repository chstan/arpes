# arpes.xarray\_html\_repr module

\# For now, this is not in use because I am going to see what I can get
done with the .S \# accessors, this way we can specialize to context

from arpes.config import SETTINGS from arpes.xarray\_extensions import
ARPESDataArrayAccessor, ARPESDatasetAccessor

original\_mods = {}

  - def patch\_method(method, name=None):
    
      - if name is None:  
        name = method.\_\_name\_\_

  - def unpatch\_method(method, name=None):
    
      - if name is None:  
        name = method.\_\_name\_\_

  - if SETTINGS.get(‘xarray\_repr\_mod’):  
    pass

  - else:  
    pass

  - def repr\_html\_arpes(self):
    
      - return {  
        ‘a’: 5, ‘b’: 8
    
    }

\#ARPESDataArrayAccessor.\_[repr\_html]() = repr\_html\_arpes
ARPESDatasetAccessor.\_[repr\_html]() = repr\_html\_arpes
