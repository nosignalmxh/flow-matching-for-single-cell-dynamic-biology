from __future__ import annotations

import ast
import json
from pathlib import Path


def _code_tree(notebook_path: str):
    nb = json.loads(Path(notebook_path).read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )
    return ast.parse(source)


def _source_tree(path: str):
    return ast.parse(Path(path).read_text())


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing function {name}")


def test_ch02_figure_helpers_preserve_padding_and_display_return_contracts():
    tree = _source_tree("src/transport_reporting.py")

    save_fig_both = _function(tree, "save_fig_both")
    calls = [
        node
        for node in ast.walk(save_fig_both)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "save_figure_formats"
    ]
    assert len(calls) == 1
    pad_keywords = [kw for kw in calls[0].keywords if kw.arg == "pad_inches"]
    assert pad_keywords and isinstance(pad_keywords[0].value, ast.Name)
    assert pad_keywords[0].value.id == "pad_inches"

    kw_defaults = dict(zip([arg.arg for arg in save_fig_both.args.kwonlyargs], save_fig_both.args.kw_defaults))
    pad_default = kw_defaults["pad_inches"]
    assert isinstance(pad_default, ast.Constant)
    assert pad_default.value == 0.1

    show_saved_png = _function(tree, "show_saved_png")
    assert not any(isinstance(node, ast.Return) for node in ast.walk(show_saved_png))


def test_ch05_display_figure_output_lives_in_src_and_returns_displayed_path():
    tree = _source_tree("src/perturbation_reporting.py")
    display_figure_output = _function(tree, "display_figure_output")
    returns = [node for node in ast.walk(display_figure_output) if isinstance(node, ast.Return)]
    assert len(returns) == 1
