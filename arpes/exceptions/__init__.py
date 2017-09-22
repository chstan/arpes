class AnalysisError(Exception):
    """
    Base class to indicate that something scientific went wrong and
    was not handled in an appropriate way.

    # Examples

    1. A bad fit from scipy.optimize in an internal function or analysis
    routine that could not be handled by the user
    """
    pass

class UnimplementedException(Exception):
    """
    Stub for unimplemented code
    """
    pass


class AnalysisWarning(UserWarning):
    """
    Non-fatal, but the user probably forgot something really significant that
    invalidates any science that follows
    """
    pass
