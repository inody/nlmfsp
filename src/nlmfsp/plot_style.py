from __future__ import annotations

import matplotlib as mpl
import shutil


_PAPER_STYLE_APPLIED = False


def apply_paper_style() -> None:
    global _PAPER_STYLE_APPLIED
    if _PAPER_STYLE_APPLIED:
        return

    use_tex = shutil.which("latex") is not None

    mpl.rcParams.update(
        {
            "text.usetex": use_tex,
            "text.latex.preamble": (
                r"\usepackage{amsmath}\usepackage{amssymb}" if use_tex else ""
            ),
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "figure.titlesize": 16,
            "lines.linewidth": 2.0,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
    _PAPER_STYLE_APPLIED = True
