from dataclasses import dataclass, field
from typing import Callable, List, Any

__all__ = ["ComposeBoth", "ReversibleLambda"]


class Identity:
    def encodes(self, x):
        return x

    def __call__(self, x):
        return x

    def decodes(self, x):
        return x

    def __repr__(self):
        return "Identity()"


_identity = Identity()


@dataclass
class ReversibleLambda:
    encodes: Callable = field(repr=False)
    decodes: Callable = field(default=lambda x: x, repr=False)

    def __call__(self, value):
        return self.encodes(value)


@dataclass
class ComposeBoth:
    """
    Meant to be virtually identical to torchvision.transforms.Compose
    except that it takes two arguments, the input and the target.
    """

    transforms: List[Any]

    def __post_init__(self):
        safe_transforms = []
        for t in self.transforms:
            if isinstance(
                t,
                (
                    tuple,
                    list,
                ),
            ):
                xt, yt = t
                t = [xt or _identity, yt or _identity]

            safe_transforms.append(t)

        self.original_transforms = self.transforms
        self.transforms = safe_transforms

    def __call__(self, x, y):
        for t in self.transforms:
            if isinstance(t, (list, tuple)):
                xt, yt = t
                x, y = xt(x), yt(y)
            else:
                x, y = t(x, y)

        return x, y

    def decodes_target(self, y):
        """
        Pull the target back in the transform stack as far as possible
        This is necessary only for the predicted target because
        otherwise we can *always* push the ground truth target and input
        forward in the transform stack.

        This is imperfect because for some transforms we need X and Y
        in order to process the data
        """
        for t in self.transforms[::-1]:
            if isinstance(t, (list, tuple)):
                _, yt = t

                y = yt.decodes(y)
            else:
                break

        return y

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(\n\t"
            + "\n\t".join([str(t) for t in self.original_transforms])
            + "\n)"
        )