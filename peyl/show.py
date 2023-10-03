"""
Show is a module containing some gadgets for more rich display of lists
and so on in IPython.
"""
from typing import Sequence


class Row:
    """
    Row is for showing lists of rich objects in the notebook.

    If caption_fn is given, it will be called for each (index, item) pair in the list, and the
    result will be placed underneath the item.
    """
    def __init__(self, items, caption_fn = None):
        self.items = list(items)
        self.caption_fn = caption_fn

    def _repr_html_(self):
        row_css = {
            'display': 'flex',
            'flex-direction': 'row',
            'flex-wrap': 'wrap',
            'row-gap': '1em',
            'align-items': 'center',
        }
        item_css = {
            'display': 'flex',
            'flex-direction': 'column',
            'align-items': 'center',
        }
        # bracket_css = {
        #     'border': '1px solid black',
        #     'width': '0.5em',
        #     'margin-right': '1em',
        # }
        row_style, item_style = (' '.join(f'{key}: {val};' for key, val in css.items()) for css in [row_css, item_css])

        def caption_item(i, item):
            if self.caption_fn is None:
                return ''
            return '<p>' + html_repr(self.caption_fn(i, item)) + '</p>'

        return ''.join([
            f'<div style="{row_style}">',
            # f'<div style="{bracket_style}"></div>',
            ', '.join([
                f'<div style="{item_style}">' + html_repr(item) + caption_item(i, item) + '</div>'
                for i, item in enumerate(self.items)
            ]),
            '</div>',
        ])


class FramedMatrix:
    def __init__(self, rows, cols, entries):
        self.rows = rows
        self.cols = cols
        self.entries = entries

    def _repr_html_(self):
        def f(v):
            return '' if v == '0' else v
        def e(i, j):
            return self.entries[i][j] if isinstance(self.entries, Sequence) else self.entries[i, j]
        return ''.join([
            '<table>',
            # First row, with an empty cell at the start.
            '<tr>',
            '<td></td>',
            *['<td style="text-align: center">' + x._repr_html_() + '</td>' for x in self.cols],
            '</tr>',
            ''.join([
                html
                for i in range(len(self.rows))
                for html in [
                    '<tr>',
                    '<td style="text-align: center">',
                    html_repr(self.rows[i]),
                    '</td>',
                    *[
                        '<td style="text-align: center">' + f(html_repr(e(i, j), prefer_latex=True)) + '</td>'
                        for j in range(len(self.cols))
                    ],
                    '</tr>',
                ]
            ]),
            '</table>',
        ])


class Latex:
    def __init__(self, markup: str):
        self.markup = markup

    def _repr_latex_(self):
        return '$' + self.markup + '$'


def html_repr(obj, prefer_latex=False):
    if prefer_latex and hasattr(obj, '_repr_latex_'):
        return ' $ ' + obj._repr_latex_() + ' $ '

    if hasattr(obj, '_repr_html_'):
        return obj._repr_html_()
    if hasattr(obj, '_repr_latex_'):
        return obj._repr_latex_()
    if isinstance(obj, str):
        return obj
    return repr(obj)
