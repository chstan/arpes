"""
Some bespoke exceptions that can be used in control sequences and to
provide more information to the user. I (Conrad) prefer to use warnings for
the latter purpose, but there are reasons to throw these errors in a variety of circumstances.
"""

class AnalysisError(Exception):
    """
    Base class to indicate that something scientific went wrong and
    was not handled in an appropriate way.

    # Examples

    1. A bad fit from scipy.optimize in an internal function or analysis
    routine that could not be handled by the user
    """


class AnalysisWarning(UserWarning):
    """
    Non-fatal, but the user probably forgot something really significant that
    invalidates any science that follows
    """


class DataPreparationError(AnalysisError):
    """
    Indicates that the user needs to perform some data preparation step
    before the analysis can proceed.
    """


class ConfigurationError(Exception):
    """
    Indicates that the user needs to supply more configuration. This could be
    setting some directories in which to place plots, or indicating what the current
    workspace being used (i.e. where to look for datasets and data) is.
    """
