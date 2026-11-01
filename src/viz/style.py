# Shared figure style so every report figure looks the same (300 dpi, one palette).

import matplotlib as mpl

# Categorical slots 1-2 of the validated reference palette; text never uses series colours.
NON_FRAUD = "#2a78d6"
FRAUD = "#eb6834"
CLASS_COLORS = {0: NON_FRAUD, 1: FRAUD}
CLASS_LABELS = {0: "Non-fraud", 1: "Fraud"}
SINGLE = NON_FRAUD
MUTED = "#86b6ef"  # lighter step of the same hue, e.g. an incomplete period
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
DPI = 300


def apply():
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )
