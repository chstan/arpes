"""Import many useful standard tools."""
from arpes.feature_gate import Gates, failing_feature_gates
from .annotations import *

from .bands import *
from .basic import *
from .dispersion import *
from .fermi_edge import *
from .fermi_surface import *
from .dos import *
from .parameter import *

from .stack_plot import *

from .spin import *
from .spatial import *

from .movie import *

# 'Tools'
# Note, we lift Bokeh imports into definitions in case people don't want to install Bokeh
# and also because of an undesirable interaction between pytest and Bokeh due to Bokeh's use
# of jinja2.
if not failing_feature_gates(Gates.LegacyUI):
    from .interactive import *
    from .band_tool import *
    from .comparison_tool import *
    from .curvature_tool import *
    from .fit_inspection_tool import *
    from .mask_tool import *
    from .path_tool import *
    from .dyn_tool import *

if not failing_feature_gates(Gates.Qt):
    from .qt_tool import qt_tool
    from .qt_ktool import ktool
    from .fit_tool import *

from .utils import savefig, remove_colorbars, fancy_labels
