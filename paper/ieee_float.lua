-- Convert pandoc tables and figures into full-width spanning floats for
-- two-column IEEEtran. Pandoc emits `longtable`, which is illegal in a
-- two-column body; we re-render each table as a `table*` + `tabular`, and
-- promote each figure to `figure*` so wide content spans both columns.

local function render(el)
  return pandoc.write(pandoc.Pandoc({ el }), "latex")
end

function Table(tbl)
  local ltx = render(tbl)
  -- longtable -> tabular
  ltx = ltx:gsub("\\begin{longtable}%[%]{@{}(.-)@{}}", "\\begin{tabular}{%1}")
  ltx = ltx:gsub("\\begin{longtable}%[%]{(.-)}", "\\begin{tabular}{%1}")
  ltx = ltx:gsub("\\begin{longtable}{(.-)}", "\\begin{tabular}{%1}")
  ltx = ltx:gsub("\\end{longtable}", "\\end{tabular}")
  -- strip longtable-only control rows
  ltx = ltx:gsub("\\endhead", "")
  ltx = ltx:gsub("\\endfirsthead", "")
  ltx = ltx:gsub("\\endlastfoot", "")
  ltx = ltx:gsub("\\endfoot", "")
  ltx = ltx:gsub("\\noalign{}", "")
  -- move the rule: pandoc puts \bottomrule right after the header block; drop
  -- all \bottomrule and add a single one before \end{tabular}
  ltx = ltx:gsub("\\bottomrule", "")
  ltx = ltx:gsub("\\end{tabular}", "\\bottomrule\n\\end{tabular}")
  local out = "\\begin{table*}[t]\n\\centering\n\\footnotesize\n"
    .. ltx
    .. "\n\\end{table*}"
  return pandoc.RawBlock("latex", out)
end

function Figure(fig)
  local ltx = render(fig)
  -- Pandoc emits a plain \begin{figure}; promote to the full-width starred float.
  -- ({ and } are literals in Lua patterns, so this matches exactly.)
  ltx = ltx:gsub("\\begin{figure}", "\\begin{figure*}[t]")
  ltx = ltx:gsub("\\end{figure}", "\\end{figure*}")
  return pandoc.RawBlock("latex", ltx)
end
