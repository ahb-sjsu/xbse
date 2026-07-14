"""Build an IEEEtran (two-column journal) PDF for IEEE TCSS submission.

Differences from the one-column build_pdf.py:
  * documentclass IEEEtran, classoption journal (two-column).
  * The "## Abstract" section is lifted into the `abstract` metadata field so it
    renders inside IEEEtran's \\begin{abstract} environment, and a keywords block
    is injected right after it.
  * Manual section numbers ("## 1.", "### 5.1") are stripped so IEEE auto-numbers
    (--number-sections).
  * A Lua filter (ieee_float.lua) rewrites pandoc's `longtable` tables as
    `table*`+`tabular` and promotes figures to `figure*`, because longtable is
    illegal in a two-column body and wide floats must span both columns.
  * Font stays Cambria (proven glyph coverage: θ β λ ≈ → ⊂ × ± …); switch mainfont
    to "TeX Gyre Termes" for an authentic Times look at camera-ready if desired.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "moral_spectrum_analyzer.md")
BUILD = os.path.join(HERE, "_build_ieee.md")
OUT = os.path.join(HERE, "moral_spectrum_analyzer_ieee.pdf")
LUA = os.path.join(HERE, "ieee_float.lua")

KEYWORDS = (
    "moral measurement, measurement invariance, content moderation, "
    "cross-dataset generalization, domain-adversarial training, "
    "moral foundations, pre-registration, AI safety"
)

# Single-quoted in YAML (backslashes literal); literal 'é' so no \'e apostrophe
# collides with YAML single-quote escaping. Pandoc parses the author metadata as
# Markdown, so the \IEEE* macros pass through as raw LaTeX.
AUTHOR = (
    r"\IEEEauthorblockN{Andrew H. Bond}"
    r"\IEEEauthorblockA{San José State University\\"
    r"ORCID 0009-0003-2599-6158 \quad andrew.bond@sjsu.edu}"
)


def build_yaml(abstract_lines):
    indented = "\n".join("  " + ln for ln in abstract_lines)
    return (
        "---\n"
        'title: "The Moral Spectrum Analyzer: A Pre-Registered Instrument for '
        'Measuring, Validating, and Discovering Moral Dimensions in Language"\n'
        f"author: '{AUTHOR}'\n"
        "documentclass: IEEEtran\n"
        "classoption: journal\n"
        'mainfont: "Cambria"\n'
        'monofont: "Consolas"\n'
        "colorlinks: true\n"
        'linkcolor: "black"\n'
        'urlcolor: "blue"\n'
        "abstract: |\n"
        f"{indented}\n"
        "header-includes: |\n"
        r"  \usepackage{graphicx}"
        "\n"
        r"  \usepackage{booktabs}"
        "\n"
        r"  \usepackage{array}"
        "\n"
        # Our Lua filter turns every Table into a RawBlock, so pandoc no longer
        # detects tables and omits the calc package + \real macro that its own
        # p{(\linewidth - ..) * \real{..}} column widths need. Supply them.
        r"  \usepackage{calc}"
        "\n"
        r"  \providecommand{\real}[1]{#1}"
        "\n"
        r"  \setlength{\tabcolsep}{4pt}"
        "\n"
        r"  \setlength{\emergencystretch}{3em}"
        "\n"
        r"  \usepackage[htt]{hyphenat}"
        "\n"
        "---\n\n"
    )


def main():
    with open(SRC, encoding="utf-8") as f:
        text = f.read()

    lines = text.splitlines()

    # Locate the two rule lines that bracket the front matter:
    #   line of first '---'  -> end of title block, start of "## Abstract"
    #   next '---'           -> end of abstract
    rules = [i for i, ln in enumerate(lines) if ln.strip() == "---"]
    first_rule, second_rule = rules[0], rules[1]

    # Abstract = everything between "## Abstract" and the second rule, minus the
    # heading line and surrounding blanks.
    abs_block = lines[first_rule + 1:second_rule]
    abs_block = [ln for ln in abs_block if ln.strip() != "## Abstract"]
    while abs_block and not abs_block[0].strip():
        abs_block.pop(0)
    while abs_block and not abs_block[-1].strip():
        abs_block.pop()

    body_lines = lines[second_rule + 1:]
    body = "\n".join(body_lines)

    # Strip manual section numbers from ATX headings so IEEE auto-numbers.
    #   "## 1. Introduction"       -> "## Introduction"
    #   "### 2.1 The channel ..."  -> "### The channel ..."
    body = re.sub(r"(?m)^(#{2,6})\s+\d+(?:\.\d+)*\.?\s+", r"\1 ", body)

    # Soften dingbats; the words carry the meaning.
    body = body.replace("✓", "").replace("✗", "")

    # IEEE keywords block, placed right after the abstract.
    keywords_block = (
        "\\begin{IEEEkeywords}\n" + KEYWORDS + "\n\\end{IEEEkeywords}\n\n"
    )

    out = build_yaml(abs_block) + keywords_block + body
    with open(BUILD, "w", encoding="utf-8") as f:
        f.write(out)

    cmd = [
        "pandoc", BUILD, "-o", OUT,
        "--pdf-engine=xelatex",
        "--lua-filter", LUA,
        "--number-sections",
        "--wrap=preserve",
        "--citeproc",
        "--bibliography", os.path.join(HERE, "references.bib"),
        "--csl", os.path.join(HERE, "ieee.csl"),
    ]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-3000:] if r.stdout else "")
    sys.stderr.write(r.stderr[-6000:] if r.stderr else "")
    if r.returncode == 0 and os.path.exists(OUT):
        print(f"\nOK -> {OUT} ({os.path.getsize(OUT)} bytes)")
    else:
        print(f"\nFAILED rc={r.returncode}")
        sys.exit(r.returncode or 1)


if __name__ == "__main__":
    main()
