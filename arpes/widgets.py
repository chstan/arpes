import matplotlib
import matplotlib.pyplot as plt

__all__ = ('pick_rectangles', 'pick_points',)


def pick_rectangles(data, **kwargs):
    ctx = {'points': [], 'rect_next': False}
    rects = []

    fig = plt.figure()
    data.S.plot(**kwargs)
    ax = fig.gca()

    def onclick(event):
        ctx['points'].append([event.xdata, event.ydata])
        if ctx['rect_next']:
            p1, p2 = ctx['points'][-2], ctx['points'][-1]
            p1[0], p2[0] = min(p1[0], p2[0]), max(p1[0], p2[0])
            p1[1], p2[1] = min(p1[1], p2[1]), max(p1[1], p2[1])

            rects.append([p1, p2])
            rect = plt.Rectangle((p1[0], p1[1],), p2[0] - p1[0], p2[1] - p1[1],
                                 edgecolor='red', linewidth=2, fill=False)
            ax.add_patch(rect)

        ctx['rect_next'] = not ctx['rect_next']
        plt.draw()

    cid = plt.connect('button_press_event', onclick)

    return rects


def pick_gamma(data, **kwargs):
    fig = plt.figure()
    data.S.plot(**kwargs)

    ax = fig.gca()
    dims = data.dims

    def onclick(event):

        data.attrs['symmetry_points'] = {
            'G': {}
        }

        print(event.x, event.xdata, event.y, event.ydata)

        for dim, value in zip(dims, [event.ydata, event.xdata]):
            if dim == 'eV':
                continue

            data.attrs['symmetry_points']['G'][dim] = value

        plt.draw()

    cid = plt.connect('button_press_event', onclick)

    return data


def pick_points(data, **kwargs):
    ctx = {'points': []}

    fig = plt.figure()
    data.S.plot(**kwargs)
    ax = fig.gca()

    x0, y0 = ax.transAxes.transform((0, 0))  # lower left in pixels
    x1, y1 = ax.transAxes.transform((1, 1))  # upper right in pixes
    dx = x1 - x0
    dy = y1 - y0
    maxd = max(dx, dy)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    width = .03 * maxd / dx * (xlim[1] - xlim[0])
    height = .03 * maxd / dy * (ylim[1] - ylim[0])

    def onclick(event):
        ctx['points'].append([event.xdata, event.ydata])

        circ = matplotlib.patches.Ellipse((event.xdata, event.ydata,), width, height, color='red')
        ax.add_patch(circ)

        plt.draw()

    cid = plt.connect('button_press_event', onclick)

    return ctx['points']