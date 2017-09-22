import numpy


def derivative(f, arg_idx=0):
    """
    Defines a simple midpoint derivative
    """

    def d(*args):
        args = list(args)
        ref_arg = args[arg_idx]
        d = ref_arg / 100

        args[arg_idx] = ref_arg + d
        high = f(*args)
        args[arg_idx] = ref_arg - d
        low = f(*args)

        return (high - low) / (2 * d)

    return d


def polarization(up, down):
    return (up - down) / (up + down)


def propagate_statistical_error(f):
    def compute_propagated_error(*args):
        running_sum = 0
        for i, arg in enumerate(args):
            df_darg_i = derivative(f, i)
            running_sum += df_darg_i(*args) ** 2 * arg

        return numpy.sqrt(running_sum)

    return compute_propagated_error
