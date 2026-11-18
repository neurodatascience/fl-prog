#!/usr/bin/env python
"""Slide figures: model forward pass and parameter legend.

Rendered with matplotlib mathtext (no LaTeX install needed). Mirrors
``LogisticRegressionModelWithShift.forward`` in ``fl_prog/model.py``.

Mathtext has no ``\\color``, so strings containing math are drawn glyph by glyph as
paths: bold glyphs of model parameters (and their bold subscripts) get the
parameter's colour, everything else the default colour.
"""

import unicodedata
from collections import defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextToPath
from matplotlib.transforms import Affine2D

from fl_prog.plotting import PARAM_COLORS, PARAM_SYMBOLS, save_fig
from fl_prog.utils.constants import CLICK_CONTEXT_SETTINGS
from fl_prog.utils.io import DEFAULT_DPATH_RESULTS

DEFAULT_DPATH_OUT = DEFAULT_DPATH_RESULTS / "figures" / "model"

DPI = 300
FONTSET = "cm"  # Computer Modern, LaTeX look
COLOR_TEXT = "#222222"
COLOR_MUTED = "#666666"
COLOR_POPULATION = "#1c4f8f"  # header colour, group-level parameters (cold)
COLOR_INDIVIDUAL = "#b8450f"  # header colour, subject-level parameters (warm)

SUBSCRIPTS = {"i", "j"}  # bold subscripts take the colour of the parameter before them

FS_EQ = 26
FS_HEADER = 15
FS_LEGEND = 14

# parameters, in bold (colours in PARAM_COLORS)
K = PARAM_SYMBOLS["k"]
M = PARAM_SYMBOLS["m"]
S = PARAM_SYMBOLS["s"]
SIGMA = PARAM_SYMBOLS["sigma"]
TAU = PARAM_SYMBOLS["tau"]
ALPHA = PARAM_SYMBOLS["alpha"]

B = r"b_{i,j}(t)"
EPS = r"\varepsilon_{i,j}(t)"
PHI = r"\phi_i(t)"

# (left-hand side, relation + right-hand side): drawn with the relations aligned
EQ_FORWARD = (
    f"${B}$",
    (
        rf"$\;=\; \dfrac{{{S}}}{{1 + \exp\left(-{K} \left("
        rf"{PHI} - {M}\right)\right)}} \;+\; {EPS}$"
    ),
)
EQ_TIME = (f"${PHI}$", rf"$\;=\; {ALPHA}\, t + {TAU}$")
EQ_NOISE = (f"${EPS}$", rf"$\;\sim\; \mathcal{{N}}\left(0,\, {SIGMA}^2\right)$")

# (symbol, description) per legend group
LEGEND_POPULATION = [
    (f"${K}$", "steepness (rate of change)"),
    (f"${M}$", "midpoint (disease time at half max)"),
    (f"${S}$", "scaling (plateau / max value)"),
    (f"${SIGMA}$", "noise standard deviation"),
]
LEGEND_INDIVIDUAL = [
    (f"${TAU}$", "time shift (disease stage at baseline)"),
    (f"${ALPHA}$", r"acceleration (progression speed, $1$ = average)"),
    (f"${PHI}$", "disease time (individual time reparametrization)"),
]
LEGEND_INDICES = [
    (r"$i$", "participant"),
    (r"$j$", "biomarker"),
    (r"$t$", "visit time"),
    (f"${B}$", "observed biomarker value"),
    (f"${EPS}$", "measurement noise"),
]

FIG_WIDTH = 12  # inches
CANVAS_HEIGHT = 20  # inches; cropped to content when saving (bbox_inches="tight")
LINE = 0.4  # inches between legend rows
EQ_LINE = 0.75  # inches between baselines of single-line equations
EQ_RELATION_X = 0.14  # figure fraction where the =/~ signs of the equations line up
HEADER_GAP = 0.5  # inches between a header and the top of its first row
ROW_BASELINE = 0.17  # inches from the top of a row to its text baseline

_TEXT_TO_PATH = TextToPath()


def _param_name(char: str) -> str:
    """Key in PARAM_COLORS for a glyph: Greek letters by Unicode name (e.g. sigma)."""
    name = unicodedata.name(char, "")
    prefix = "GREEK SMALL LETTER "
    return name.removeprefix(prefix).lower() if name.startswith(prefix) else char


def _glyph_colors(glyphs, default_color):
    """Colour of each mathtext glyph: parameter colour for bold parameter glyphs."""
    colors = []
    last_param_color = default_color  # glyphs come in layout order: base, then scripts
    for font, _, ccode, _, _, _ in glyphs:
        char = unicodedata.normalize("NFKC", chr(ccode))
        is_bold = "bold" in font.postscript_name.lower() or "BOLD" in unicodedata.name(
            chr(ccode), ""
        )
        color = default_color
        if is_bold and _param_name(char) in PARAM_COLORS:
            color = last_param_color = PARAM_COLORS[_param_name(char)]
        elif is_bold and char in SUBSCRIPTS:
            color = last_param_color
        colors.append(color)
    return colors


def math_paths(s, color):
    """Paths (in FONT_SCALE pt units, baseline at y=0) grouped by colour, and width."""
    prop = FontProperties(size=_TEXT_TO_PATH.FONT_SCALE)
    width, _, _, glyphs, _ = _TEXT_TO_PATH.mathtext_parser.parse(
        s, _TEXT_TO_PATH.DPI, prop
    )
    glyph_info, glyph_map, rects = _TEXT_TO_PATH.get_glyphs_mathtext(prop, s)
    assert len(glyph_info) == len(glyphs)

    paths = defaultdict(lambda: ([], []))
    for (glyph_repr, x, y, scale), glyph_color in zip(
        glyph_info, _glyph_colors(glyphs, color)
    ):
        verts, codes = glyph_map[glyph_repr]
        paths[glyph_color][0].extend(np.asarray(verts) * scale + [x, y])
        paths[glyph_color][1].extend(codes)
    for verts, codes in rects:  # fraction bars etc.
        paths[color][0].extend(verts)
        paths[color][1].extend(codes)
    return {c: MplPath(np.asarray(v), codes) for c, (v, codes) in paths.items()}, width


class Canvas:
    """Place text at x (figure fraction) and y (inches from the top of the figure)."""

    def __init__(self):
        self.fig = plt.figure(figsize=(FIG_WIDTH, CANVAS_HEIGHT))

    def text(
        self, x, y_in, s, fontsize, color=COLOR_TEXT, ha="left", va="top", **kwargs
    ):
        if "$" not in s:
            self.fig.text(
                x, 1 - y_in / CANVAS_HEIGHT, s, fontsize=fontsize, color=color,
                ha=ha, va=va, **kwargs,
            )  # fmt: skip
            return

        paths, width = math_paths(s, color)
        scale = fontsize / _TEXT_TO_PATH.FONT_SCALE  # path units -> points
        x_pt = (
            x * FIG_WIDTH * 72
            - {"left": 0, "center": width / 2, "right": width}[ha] * scale
        )
        y_pt = (CANVAS_HEIGHT - y_in) * 72
        if va == "top":
            y_pt -= max(path.vertices[:, 1].max() for path in paths.values()) * scale
        # points -> figure fraction (transFigure, unlike dpi_scale_trans, follows
        # the crop done by bbox_inches="tight")
        transform = (
            Affine2D()
            .scale(scale)
            .translate(x_pt, y_pt)
            .scale(1 / (72 * FIG_WIDTH), 1 / (72 * CANVAS_HEIGHT))
            + self.fig.transFigure
        )
        for path_color, path in paths.items():
            self.fig.add_artist(
                PathPatch(
                    path, facecolor=path_color, edgecolor="none", transform=transform
                )
            )

    def header(self, x, y_in, s, color=COLOR_TEXT):
        self.text(x, y_in, s, fontsize=FS_HEADER, fontweight="bold", color=color)
        return y_in + HEADER_GAP

    def legend_group(
        self, x, y_in, title, title_color, items, symbol_color, symbol_width=0.075
    ):
        """Titled list of (symbol, description) rows; return y below the group."""
        y_in = self.header(x, y_in, title, title_color) + ROW_BASELINE
        for symbol, description in items:
            self.text(
                x,
                y_in,
                symbol,
                fontsize=FS_LEGEND + 2,
                color=symbol_color,
                va="baseline",
            )
            self.text(
                x + symbol_width, y_in, description, fontsize=FS_LEGEND, va="baseline"
            )
            y_in += LINE
        y_in -= ROW_BASELINE
        return y_in


def draw_forward(canvas: Canvas, y_in=0.0):
    """Forward-pass equation, time reparametrization and noise model, one per line."""
    # (equation, inches from previous baseline); the fraction line needs extra room
    lines = (
        (EQ_FORWARD, 0.65),
        (EQ_TIME, EQ_LINE + 0.3),
        (EQ_NOISE, EQ_LINE),
    )
    for (lhs, rhs), gap in lines:
        y_in += gap
        canvas.text(EQ_RELATION_X, y_in, lhs, fontsize=FS_EQ, ha="right", va="baseline")
        canvas.text(EQ_RELATION_X, y_in, rhs, fontsize=FS_EQ, ha="left", va="baseline")
    return y_in + 0.45


def draw_legend(canvas: Canvas, y_in=0.0):
    """Parameter legend in two columns; return y below the content."""
    y_left = canvas.legend_group(
        0.03,
        y_in,
        "Population parameters (shared across sites)",
        COLOR_POPULATION,
        LEGEND_POPULATION,
        COLOR_TEXT,
    )
    y_right = canvas.legend_group(
        0.53,
        y_in,
        "Individual parameters (stay at each site)",
        COLOR_INDIVIDUAL,
        LEGEND_INDIVIDUAL,
        COLOR_TEXT,
    )
    y_right = canvas.legend_group(
        0.53,
        y_right + 0.25,
        "Indices and data",
        COLOR_MUTED,
        LEGEND_INDICES,
        COLOR_MUTED,
    )
    return max(y_left, y_right)


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--output-dir",
    "dpath_out",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_DPATH_OUT,
    show_default=True,
    help="Output directory (PNG for forward pass, legend and combined figure).",
)
@click.option("--transparent", is_flag=True, help="Transparent background.")
def plot_presentation_model(dpath_out: Path, transparent: bool):
    plt.rcParams["mathtext.fontset"] = FONTSET
    plt.rcParams["font.family"] = "DejaVu Sans"

    figures = {}
    for fname, draw in (
        ("model_forward", draw_forward),
        ("model_legend", draw_legend),
    ):
        canvas = Canvas()
        draw(canvas)
        figures[fname] = canvas.fig

    canvas = Canvas()
    y_in = draw_forward(canvas)
    draw_legend(canvas, y_in + 0.3)
    figures["model_overview"] = canvas.fig

    for fname, fig in figures.items():
        save_fig(fig, dpath_out / f"{fname}.png", dpi=DPI, transparent=transparent)
        plt.close(fig)


if __name__ == "__main__":
    plot_presentation_model()
