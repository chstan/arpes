import pyqtgraph as pg

from .data_array_image_view import DataArrayImageView
from .help_dialogs import BasicHelpDialog
from .windows import SimpleWindow
from .app import SimpleApp

__all__ = ('setup_pyqtgraph', 'DataArrayImageView',
           'BasicHelpDialog', 'SimpleWindow', 'SimpleApp', 'qt_info',)


class QtInfo:
    screen_dpi = 150

    def __init__(self):
        self._inited = False
        self._pg_patched = False

    def init_from_app(self, app):
        if self._inited:
            return

        self._inited = True
        screen = app.screens()[0]
        dpis = [screen.physicalDotsPerInch() for screen in app.screens()]
        self.screen_dpi = sum(dpis) / len(dpis)

    def inches_to_px(self, arg):
        if isinstance(arg, (int, float,)):
            return self.screen_dpi * arg

        return map(lambda x: x * self.screen_dpi, arg)

    def setup_pyqtgraph(self):
        """
        Does any patching required on PyQtGraph and configures options.
        :return:
        """

        if self._pg_patched:
            return

        self._pg_patched = True

        pg.setConfigOptions(antialias=True, foreground=(0, 0, 0), background=(255, 255, 255))

        def patchedLinkedViewChanged(self, view, axis):
            """
            This still isn't quite right but it is much better than before. For some reason
            the screen coordinates of the PlotWidget are not being computed correctly, so
            we will just lock them as though they were perfectly aligned.

            This will clearly not work well for plots that have to be coordinated across
            different parts of the layout, but this will work for now.

            We also don't handle inverted axes for now.
            :param self:
            :param view:
            :param axis:
            :return:
            """
            if self.linksBlocked or view is None:
                return

            vr = view.viewRect()
            vg = view.screenGeometry()
            sg = self.screenGeometry()
            if vg is None or sg is None:
                return

            view.blockLink(True)

            try:
                if axis == pg.ViewBox.XAxis:
                    upp = float(vr.width()) / vg.width()
                    overlap = min(sg.right(), vg.right()) - max(sg.left(), vg.left())

                    if overlap < min(vg.width() / 3, sg.width() / 3):
                        x1 = vr.left()
                        x2 = vr.right()
                    else:  # attempt to align
                        x1 = vr.left()
                        x2 = vr.right() + (sg.width() - vg.width()) * upp

                    self.enableAutoRange(pg.ViewBox.XAxis, False)
                    self.setXRange(x1, x2, padding=0)
                else:
                    upp = float(vr.height()) / vg.height()
                    overlap = min(sg.bottom(), vg.bottom()) - max(sg.top(), vg.top())

                    if overlap < min(vg.height() / 3, sg.height() / 3):
                        y1 = vr.top()
                        y2 = vr.bottom()
                    else:  # again, attempt to align
                        y1 = vr.top()  # snap them at one side to the same coordinate

                        # and scale the other side
                        y2 = vr.bottom() + (sg.height() - vg.height()) * upp

                    self.enableAutoRange(pg.ViewBox.YAxis, False)
                    self.setYRange(y1, y2, padding=0)
            finally:
                view.blockLink(False)

        pg.ViewBox.linkedViewChanged = patchedLinkedViewChanged


qt_info = QtInfo() # singleton configuration

