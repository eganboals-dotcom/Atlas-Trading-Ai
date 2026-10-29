"""LinkedIn document post (a swipeable PDF carousel) from the write-up's charts: docs/report/atlas_carousel.pdf.

Run after scripts/build_report.py; reads docs/report/figures/*.png and data.json. Pages are 4:5 portrait (LinkedIn's
best-filling shape on phones).

    .venv/Scripts/python.exe scripts/build_carousel.py
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

FIG = Path("docs/report/figures")
OUT = Path("docs/report/atlas_carousel.pdf")
REPO = "github.com/eganboals-dotcom/Atlas-Trading-Ai"
SURFACE, INK, INK2, MUTED, ACCENT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#2a78d6"
W, H = 8.0, 10.0  # inches, 4:5
CROP = {"4_scoreboard.png": (0.115, 0.03), "6_years_to_prove.png": (0.185, 0.045)}  # (top, bottom) fractions cut off each chart; default below
PREVIEW = Path(sys.argv[1]) if len(sys.argv) > 1 else None  # optional folder for PNG previews of each page

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "text.parse_math": False})


def page(pdf: PdfPages, n: int, total: int, kicker: str, headline: str, body: str = "", image: str | None = None,
         big: str | None = None) -> None:
    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    fig.text(0.07, 0.94, kicker.upper(), fontsize=12, color=ACCENT, fontweight="bold", va="top")
    fig.text(0.07, 0.90, headline, fontsize=25, color=INK, fontweight="bold", va="top", wrap=True, linespacing=1.15)
    if big:
        fig.text(0.07, 0.62, big, fontsize=64, color=ACCENT, fontweight="bold", va="center")
    if image:  # the slide headline replaces the chart's own title: crop it (and the source line) off
        img = mpimg.imread(FIG / image)
        top, bottom = CROP.get(image, (0.21, 0.045))
        img = img[int(len(img) * top): int(len(img) * (1 - bottom))]
        w = 0.94
        h = w * W * img.shape[0] / img.shape[1] / H
        if h > 0.58:  # tall charts (the scoreboard): fit the height and center
            h = 0.58
            w = h * H * img.shape[1] / img.shape[0] / W
        ax = fig.add_axes([(1 - w) / 2, 0.80 - h, w, h])
        ax.imshow(img)
        ax.axis("off")
        y_body = 0.80 - h - 0.03
    else:
        y_body = 0.74
    if body:
        wrapped = "\n".join(textwrap.fill(par, 58) if par else "" for par in body.split("\n"))
        fig.text(0.07, y_body, wrapped, fontsize=15 if image else 18.5, color=INK2, va="top", linespacing=1.45)
    fig.text(0.07, 0.035, "Atlas: research and paper trading only. Not investment advice.", fontsize=9.5, color=MUTED)
    fig.text(0.93, 0.035, f"{n}/{total}", fontsize=10, color=MUTED, ha="right")
    pdf.savefig(fig, facecolor=SURFACE)
    if PREVIEW:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / f"{n}.png", dpi=80, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    d = json.loads((FIG / "data.json").read_text(encoding="utf-8"))
    ov, costs, sb = d["overfitting"], d["costs"], d["scoreboard"]
    hold = sb["hold"]["ann_return"]
    slides = [
        dict(kicker="A trading research project", headline=f"I tested {ov['total_trials'] // 1000:,},000 trading bots.\n"
             "Almost none beat simply\nbuying and holding.",
             body="Ten years of US stock data, a live paper-trading desk, and every result judged on days the bots "
                  "had never seen, after trading costs.\n\nHere's why beating the market is so hard.  >>",
             big=None),
        dict(kicker="1. Backtests flatter", headline=f"Training scores overstated the\nresult in {ov['worse']} of "
             f"{len(ov['rows'])} experiments",
             body="Search through enough strategies and some fit the past by pure luck. The better the best one looks, "
                  "the less that number means.", image="1_backtest_vs_unseen.png"),
        dict(kicker="2. Costs", headline="The trading bots lost almost\nexactly what they paid to trade",
             body=f"On live prices they paid ${costs['paid_trading_bots']:,.0f} in spreads and lost "
                  f"${costs['lost_trading_bots']:,.0f}. Before costs, they roughly broke even. The other side of "
                  "those trades (market makers) earns the spread.", image="3_before_after_costs.png"),
        dict(kicker="3. Textbook strategies", headline=f"Only {sb['beat']} of {len(sb['rows'])} strategies beat\n"
             "holding all 300 stocks",
             body=f"Hold earned {hold:+.1%} a year. The two biggest winners held just 3 stocks at a time and fell 54% "
                  "at their worst: one lucky run, not a method.", image="4_scoreboard.png"),
        dict(kicker="4. Signals", headline="'Good' stocks didn't reliably\nbeat 'bad' ones",
             body="Each signal splits the stocks into fifths. Several famous ones sorted them the wrong way on "
                  "the years they hadn't seen.", image="5_best_vs_worst_fifth.png"),
    ]
    if (FIG / "6_years_to_prove.png").exists():
        slides.append(dict(kicker="5. Tiny edges", headline="The edges that survived take\ndecades to prove",
                           body="The cautious versions of the rules that passed beat holding by under a point a year. Telling "
                                "that apart from luck takes decades of data, and published edges shrink once people "
                                "trade them.",
                           image="6_years_to_prove.png"))
    slides += [
        dict(kicker="So who makes money?", headline="Mostly the other side\nof the trade",
             body="- Market makers, who earn the spread everyone else pays\n"
                  "- Brokers, through commissions and order flow\n"
                  "- Very few day traders: in Brazil, ~97% of persistent day traders lost money; in Taiwan, under 1% "
                  "were reliably profitable\n"
                  "- Most professional funds trail their index over 10-15 years"),
        dict(kicker="What's next", headline="The survivors now face\nthe only honest test",
             body="Six frozen trackers score every surviving rule on real future days, against buy-and-hold. Their "
                  "rules are fingerprinted, so nothing can be tuned after the fact.\n\n"
                  f"Full write-up, interactive models and code:\n{REPO}"),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        for i, s in enumerate(slides, 1):
            page(pdf, i, len(slides), **s)
    print("wrote", OUT, len(slides), "pages")


if __name__ == "__main__":
    main()
