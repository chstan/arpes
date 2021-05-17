from typing import Dict, Any

__all__ = [
    "SpectrumFormatter",
    "FloatTitleFormatter",
]


class SpectrumFormatter:
    def show(self, data, ax=None):
        spectrum, row = data
        ax.imshow(spectrum, origin="lower")


class FloatTitleFormatter:
    context: Dict[str, Any] = None
    title_formatter: str = r"{label}={data:.3f}"

    @property
    def computed_context(self) -> Dict[str, Any]:
        return {
            "label": "True" if self.context.get("is_ground_truth", False) else "Pred"
        }

    def show(self, data, ax=None):
        title = ax.get_title()
        context = {
            **self.computed_context,
            "data": data,
        }

        ax.set_title(
            "{title}; {addendum}".format(
                title=title, addendum=self.title_formatter.format(**context)
            )
        )
