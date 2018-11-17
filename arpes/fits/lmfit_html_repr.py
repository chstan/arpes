"""
For now we monkeypatch lmfit to make it easier to work with in Jupyter. We should consider forking
or providing a pull at a later date after this settles down.
"""

import numpy as np
from lmfit import model


def repr_html_ModelResult(self, **kwargs):
    template = """
        <div>
            <span><strong>Converged: </strong>{success}</span>
            <div>{formatted_components}<div>
            <div>{parameters}</div>
        </div>
        """
    return template.format(
        success=self.success,
        formatted_components=''.join('<div>{}</div>'.format(c._repr_html_()) for c in self.components),
        parameters=self.params._repr_html_(**kwargs)
    )

def repr_html_Model(self):
    template = """
    <div>
    <strong>{name}</strong>
    </div>
    """
    return template.format(name=self.name)


def repr_html_Parameters(self, short=False):
    skip_on_short = {'Min', 'Max', 'Vary', 'Expr', 'Brute_Step'}
    all = ['Name', 'Value', 'Min', 'Max', 'Stderr', 'Vary', 'Expr', 'Brute_Step']
    keys = list(self.keys())
    keys.sort()
    template = """
    <table>
      <thead>
          <tr>
            {cols}
          </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    """
    return template.format(
        cols=''.join('<th>{}</th>'.format(c) for c in all if not short or c not in skip_on_short),
        rows=''.join(self[p].to_table_row(short=short) for p in keys)
    )

def repr_html_Parameter(self, short=False):
    if short:
        return """
            <tr>
                <th>{name}</th>
                <th>{value:.3f}</th>
                <th>{stderr:.3f}</th>
            </tr>
            """.format(
            name=self.name,
            value=self.value,
            stderr=self.stderr,
        )

    template = """
            <tr>
                <th>{name}</th>
                <th>{value:.3f}</th>
                <th>{min:.3f}</th>
                <th>{max:.3f}</th>
                <th>{stderr:.3f}</th>
                <th>{vary}</th>
                <th>{expr}</th>
                <th>{brute_step}</th>
            </tr>
            """
    return template.format(
        name=self.name,
        value=self.value,
        min=self.min,
        max=self.max,
        stderr=self.stderr or np.inf,
        vary=self.vary,
        expr=self.expr or '',
        brute_step=self.brute_step or ''
    )


model.Model._repr_html_ = repr_html_Model
#model.Parameter._repr_html_ = repr_html_Parameter
model.Parameters._repr_html_ = repr_html_Parameters
model.ModelResult._repr_html_ = repr_html_ModelResult

model.Parameter.to_table_row = repr_html_Parameter

__all__ = tuple()
