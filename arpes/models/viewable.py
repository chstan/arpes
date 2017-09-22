import itertools
import warnings

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy
import numpy.random
from scipy import ndimage

from arpes.utilities import arrange_by_indices


class Viewable(object):
    def axis_by_name(self, name):
        axis_index = self.axis_names.index(name)
        return self.axes[axis_index]

    def rename_axis(self, name, new_name):
        axis_index = self.axis_names.index(name)
        self.axes[axis_index]['name'] = new_name

    def offset_axis(self, name, offset):
        axis_index = self.axis_names.index(name)
        low, high, step = self.axes[axis_index]['bounds']
        self.axes[axis_index]['bounds'] = (
            low + offset,
            high + offset,
            step
        )
        self.bounds[axis_index] = self.axes[axis_index]['bounds']

    def axis_by_name(self, name):
        axis_index = self.axis_names.index(name)
        return self.axes[axis_index]

    def _build_single_transform(self, transform):
        """
        This is a bit weird, but for now this makes more sense than building
        transforms external to the Viewable, the reason being that the
        Viewable has a lot of internal information that it is usful to know
        about during the transform process. A good way to approximate this
        knowledge would be to pass around the source Viewable as well as the
        data in transit.

        Anyway, you can reassess later, Conrad.
        """
        default_smoothing_order = 8

        if transform == 'raw':
            return lambda x: x

        if 'sobel-' in transform:
            transform_axis_name = transform[len('sobel-'):]
            transform_axis = self.axis_names.index(transform_axis_name)

            def sobel_edge(data):
                return ndimage.sobel(data, transform_axis)

            return sobel_edge

        if 'prewitt-' in transform:
            transform_axis_name = transform[len('prewitt-'):]
            transform_axis = self.axis_names.index(transform_axis_name)

            def prewitt_edge(data):
                return ndimage.prewitt(data, transform_axis)

            return prewitt_edge

        if 'mf' in transform:
            # median filter smoothing
            t, t_axis_name = transform.split('-')
            if len(t) == 2:
                smoothing_order = default_smoothing_order
            else:
                smoothing_order = int(t[len('mf'):])

            def smooth(data):
                transform_axis = self.axis_names.index(t_axis_name)
                sigmas = tuple(smoothing_order if axis == t_axis_name else 1
                               for axis in self.axis_names)
                return ndimage.median_filter(data, size=sigmas)

            return smooth

        if 'gs' in transform:
            # gaussian smoothing
            t, t_axis_name = transform.split('-')

            if len(t) == 2:
                smoothing_order = default_smoothing_order
            else:
                smoothing_order = int(t[len('gs'):])

            def smooth(data):
                transform_axis = self.axis_names.index(t_axis_name)
                sigmas = tuple(smoothing_order if axis == t_axis_name else 0
                               for axis in self.axis_names)
                return ndimage.gaussian_filter(data, sigma=sigmas, order=0)

            return smooth

        if 'clz' in transform:
            if '-' in transform:
                clamp = float(transform.split('-')[1])
            else:
                clamp = 0.1

            def perform_clamp(data):
                """
                Take the raw data and flatten everything within a factor of 0
                """

                full_range = numpy.max(numpy.abs(data))
                q = numpy.copy(data)
                q[numpy.abs(q) < full_range * clamp] = 0
                return q

            return perform_clamp

        if 'clh' in transform:
            if '-' in transform:
                clamp = float(transform.split('-')[1])
            else:
                clamp = 0.9

            def perform_clamp(data):
                """
                Take the raw data and flatten the highest values
                """

                full_range = numpy.max(numpy.abs(data))
                q = numpy.copy(data)
                q[numpy.abs(q) > full_range * clamp] = 0
                return q

            return perform_clamp

        if 'd2-' in transform:
            transform_axis_name = transform[len('d2-'):]
            transform_axis = self.axis_names.index(transform_axis_name)
            def take_d2(data):
                return numpy.gradient(numpy.gradient(data, axis=transform_axis),
                                      axis=transform_axis)

            return take_d2

        # out of options!
        warnings.warn('Could not build filter for {}, returning id'.format(
            transform))

        return lambda x: x

    def get_transformed_data(self, transform):
        """
        This helper is used to fetch data in a particular format for mapping
        or slicing or manipulating in some way.

        # Common usages:

        - raw data with 'raw'
        - second derivative band mapping with 'd2-{axis_name}'
        - sobel mapping with 'sobel-{axis_name}'
        """
        if transform is None:
            transform = []

        if isinstance(transform, str):
            transform = [transform]

        data = self.data
        for t in transform:
            f = self._build_single_transform(t)
            data = f(data)

        return data

    def tslice(self, axis_order=None, make_plot=False, transform=None, **kwargs):
        """
        This is bit of an overloaded function, so I'll probably have to cut it up
        at some later point, but the idea is to provide a convenient way to slice
        into data according to the dimensionful axes attached, perform
        transformations on said data, and optionally to create plots and animations
        of these cuts.

        Curently only 0 - 3D cuts are allowed, but this is the maximum size
        of a Viewable anyway, so there's no immediate issue with this constraint.
        """
        movie_axis_name = kwargs.pop('movie', None)
        special_axes = set([movie_axis_name])

        # As a first step, we need to go through the kwargs and turn them into
        # indices
        index_dict = {axis_name: self.value_to_index(value, axis_name)
                      for axis_name, value in kwargs.items()}

        if movie_axis_name is not None and movie_axis_name not in index_dict:
            index_dict[movie_axis_name] = self.value_to_index(
                self.bounds[self.axis_names.index(movie_axis_name)][:2],
                movie_axis_name)

        partial_bounds = {k: itertools.chain(v, self.axis_by_name(k)['bounds'][2])
                          for k, v in kwargs.items() if isinstance(v, list)}

        # step through axes to build slices, bounds
        fixed_axes = set(k for k, v in kwargs.items() if not isinstance(v, list))

        def build_slice(spec):
            if isinstance(spec, list):
                return numpy.s_[spec[0]:spec[1]]
            return numpy.s_[spec]

        slices = [build_slice(index_dict[name]) if name in index_dict else numpy.s_[:]
                  for name in self.axis_names]

        free_bounds = [kwargs.get(name, self.axis_by_name(name)['bounds'][:2])
                       for name in self.axis_names
                       if name not in fixed_axes.union(special_axes)]

        free_axis_names = [name for name in self.axis_names
                           if name not in fixed_axes.union(special_axes)]

        sliced = self.get_transformed_data(transform)[slices]
        sliced_dimensions = len(sliced.shape)
        if sliced_dimensions == 3 and make_plot:
            movie_axis = self.axis_names.index(movie_axis_name)

        # Next let's consider whether it was requested that we produce the axes
        # in a particular order. If so, we need to transpose and swap labels
        # around. There's also a bit of care we need to take if we are making a
        # movie, since in that case the movie axis ends up somewhere in the middle
        # of the data and we need to avoid perturbing it
        if axis_order is not None:
            if movie_axis_name is not None:
                assert(len(axis_order) + 1 == sliced_dimensions)
                axis_order_indices = [
                    [n for n in self.axis_names if n not in fixed_axes].index(a)
                    for a in itertools.chain(axis_order, [movie_axis_name])
                ]
                free_bounds = arrange_by_indices(free_bounds, axis_order_indices[:-1])
                free_axis_names = arrange_by_indices(
                    free_axis_names, axis_order_indices[:-1])
                movie_axis = len(axis_order_indices) - 1
                sliced = numpy.transpose(sliced, axes=axis_order_indices)
            else:
                axis_order_indices = [
                    [n for n in self.axis_names if n not in fixed_axes].index(a)
                    for a in axis_order
                ]
                sliced = numpy.transpose(sliced, axes=axis_order_indices)
                free_bounds = arrange_by_indices(free_bounds, axis_order_indices)
                free_axis_names = arrange_by_indices(
                    free_axis_names, axis_order_indices)

        if make_plot:
            if sliced_dimensions == 3:
                # make a movie
                desired_time = 5000 # 5 seconds
                number_of_frames = sliced.shape[movie_axis]
                if number_of_frames > 100:
                    desired_time = 15000

                def slice_at_i(i):
                    return [numpy.s_[:] if j != movie_axis else numpy.s_[i]
                            for j in range(3)]

                fig = plt.figure()
                current_data = sliced[slice_at_i(0)]
                image = plt.imshow(current_data[::-1, :],
                                   interpolation='none', aspect='auto',
                                   extent=list(itertools.chain(*free_bounds[::-1])))
                plt.xlabel(free_axis_names[1])
                plt.ylabel(free_axis_names[0])

                def init_cut():
                    return [image]

                def animate(i):
                    cd = sliced[slice_at_i(i)]
                    image.set_data(cd[::-1, :])
                    value_at_slice = self.index_to_value(
                        index_dict[movie_axis_name][0] + i, movie_axis_name)
                    plt.title('{0}: {1:+.03g}'.format(movie_axis_name, value_at_slice))
                    fig.canvas.draw()
                    return [image]

                return animation.FuncAnimation(
                    fig, animate, init_func=init_cut, frames=number_of_frames,
                    repeat=False, interval=(desired_time//number_of_frames),
                    blit=True)

            if sliced_dimensions == 2:
                plt.imshow(sliced[::-1, :], interpolation='none', aspect='auto',
                           extent=list(itertools.chain(*free_bounds[::-1])))
                plt.xlabel(free_axis_names[1])
                plt.ylabel(free_axis_names[0])
            elif sliced_dimensions == 1:
                plt.plot(range(sliced.shape[0]), sliced)
                plt.ylabel('Spectrum Intensity')
                plt.xlabel(free_axis_names[0])
            else:
                warnings.warn('You cannot request a plot from tslice unless N == 1 or 2.')

        return sliced

    @property
    def axis_names(self):
        return [axis['name'] for axis in self.axes]

    def index_to_value(self, index, axis=None):
        """
        The approximate inverse function to 'value_to_index'
        """

        if axis is None:
            axis = self.axis_names[0]

        axis_index = self.axis_names.index(axis)
        low, high, _ = self.axis_by_name(axis)['bounds']
        n = self.data.shape[axis_index]
        return low + index * (high - low) / (n-1)

    def value_to_index(self, value, axis=None):
        if axis is None:
            axis = self.axis_names[0]

        n = self.data.shape[self.axis_names.index(axis)]
        bounds = self.axis_by_name(axis)['bounds']

        try:
            return [int(n * (v - bounds[0])/ (bounds[1] - bounds[0])) for v in value]
        except TypeError:
            return int(n * (value - bounds[0])/ (bounds[1] - bounds[0]))


    @property
    def ticks(self):
        return [numpy.linspace(b[0], b[1], s)
                for b, s in zip(self.bounds, self.data.shape)]

    def __init__(self, data=None, axes=None, bounds=None, **kwargs): #pylint: disable=W0613
        self.axes = axes
        self.bounds = bounds
        self.data = data
