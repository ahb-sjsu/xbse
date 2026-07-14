"""Build moral_spectrum_analyzer.pdf from the markdown via pandoc + xelatex.
Strips the markdown title block (YAML metadata supplies it), softens two dingbats
that a serif face may lack, and prepends a pandoc metadata header.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "moral_spectrum_analyzer.md")
BUILD = os.path.join(HERE, "_build.md")
OUT = os.path.join(HERE, "moral_spectrum_analyzer.pdf")

YAML = r"""---
title: "The Moral Spectrum Analyzer"
subtitle: "A Pre-Registered Instrument for Measuring, Validating, and Discovering Moral Dimensions in Language"
author: "Andrew H. Bond \\newline San José State University · ORCID 0009-0003-2599-6158 · andrew.bond@sjsu.edu"
date: "2026-07-13 · Draft (numbers verified against source artifacts)"
geometry: "margin=1in"
fontsize: 11pt
mainfont: "Cambria"
monofont: "Consolas"
colorlinks: true
linkcolor: "MidnightBlue"
urlcolor: "MidnightBlue"
header-includes: |
  \usepackage{setspace}
  \setstretch{1.05}
  \usepackage{float}
  \makeatletter
  \def\fps@figure{htbp}
  \makeatother
  \usepackage{sectsty}
  \sectionfont{\large}
  \subsectionfont{\normalsize}
  \usepackage{etoolbox}
  \AtBeginEnvironment{longtable}{\small}
  \AtBeginEnvironment{tabular}{\small}
  \setlength{\tabcolsep}{4pt}
  \setlength{\emergencystretch}{3em}
  \usepackage[htt]{hyphenat}
---

"""


def main():
    with open(SRC, encoding="utf-8") as f:
        text = f.read()

    # Strip the markdown title block: everything up to and including the first
    # horizontal rule (---) that follows the title/author/draft lines.
    lines = text.splitlines()
    cut = None
    for i, ln in enumerate(lines):
        if ln.strip() == "---" and i > 0:
            cut = i
            break
    body = "\n".join(lines[cut + 1:]) if cut is not None else text

    # Soften dingbats that a serif face may render as tofu; the words carry the meaning.
    body = body.replace("✓", "").replace("✗", "")

    with open(BUILD, "w", encoding="utf-8") as f:
        f.write(YAML + body)

    cmd = [
        "pandoc", BUILD, "-o", OUT,
        "--pdf-engine=xelatex",
        "--toc", "--toc-depth=2",
        "--wrap=preserve",
        "--citeproc",
        "--bibliography", os.path.join(HERE, "references.bib"),
        "--csl", os.path.join(HERE, "ieee.csl"),
    ]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-3000:] if r.stdout else "")
    sys.stderr.write(r.stderr[-4000:] if r.stderr else "")
    if r.returncode == 0 and os.path.exists(OUT):
        print(f"\nOK -> {OUT} ({os.path.getsize(OUT)} bytes)")
    else:
        print(f"\nFAILED rc={r.returncode}")
        sys.exit(r.returncode or 1)


if __name__ == "__main__":
    main()
