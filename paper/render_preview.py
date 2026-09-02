"""
render_preview.py
=================
Turn paper.tex into a readable HTML preview, with every macro substituted and
every figure embedded.

WHY THIS EXISTS
LaTeX is not installed on this machine, so `pdflatex` cannot run here. But you
still need to READ the paper -- to check the argument flows, that the numbers
read correctly in context, and that nothing is missing. This renders a
faithful-enough preview in a browser, with no installation required.

IMPORTANT: this is a PREVIEW, not a substitute for compiling. It does not
reproduce IEEE two-column formatting, exact figure placement, or proper
citation numbering. Use Overleaf (or install MiKTeX) to produce the actual
submission PDF. See paper/README.md.

Run me (from the project root):
    python paper/render_preview.py
Then open  paper/paper_preview.html  in any browser.
"""

import base64
import html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config    # noqa: E402

PAPER = os.path.join(config.PAPER_DIR, "paper.tex")
MACROS = os.path.join(config.PAPER_DIR, "results_macros.tex")
OUT = os.path.join(config.PAPER_DIR, "paper_preview.html")


# ---------------------------------------------------------------------------
def load_macros():
    """Read \\newcommand{\\Name}{value} pairs out of results_macros.tex."""
    text = open(MACROS, encoding="utf-8").read()
    return dict(re.findall(r"newcommand\{\\(\w+)\}\{([^}]*)\}", text))


def substitute_macros(tex, macros):
    """Replace \\Name{} and \\Name with their values. Longest name first, so
    \\CNNaccLow is not mangled by a shorter \\CNNacc."""
    for name in sorted(macros, key=len, reverse=True):
        val = macros[name]
        tex = tex.replace("\\" + name + "{}", val)
        tex = re.sub(r"\\" + name + r"(?![A-Za-z])", val, tex)
    return tex


def embed(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------
def convert_tabular(block):
    """Convert a LaTeX tabular body into an HTML table."""
    body = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}",
                     block, re.S)
    if not body:
        return ""
    rows_raw = body.group(1)
    rows_raw = re.sub(r"\\(top|mid|bottom)rule", "", rows_raw)
    out = ["<table>"]
    first = True
    for line in rows_raw.split(r"\\"):
        line = line.strip()
        if not line:
            continue
        cells = [inline(c.strip()) for c in line.split("&")]
        tag = "th" if first else "td"
        out.append("<tr>" + "".join(
            "<{0}>{1}</{0}>".format(tag, c) for c in cells) + "</tr>")
        first = False
    out.append("</table>")
    return "\n".join(out)


def inline(s):
    """Convert inline LaTeX markup to HTML."""
    s = re.sub(r"\\(?:textbf|textsc)\{([^{}]*)\}", r"<strong>\1</strong>", s)
    s = re.sub(r"\\(?:emph|textit)\{([^{}]*)\}", r"<em>\1</em>", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"<code>\1</code>", s)
    s = re.sub(r"\\cite\{([^}]*)\}",
               lambda m: "[" + ", ".join(
                   k.strip() for k in m.group(1).split(",")) + "]", s)
    s = re.sub(r"\\ref\{([^}]*)\}", r"<em>\1</em>", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    # maths: strip the delimiters, keep the content readable
    s = re.sub(r"\$([^$]*)\$", lambda m: "<i>" + tidy_math(m.group(1)) + "</i>", s)
    s = s.replace("---", "&mdash;").replace("--", "&ndash;")
    s = s.replace("``", "&ldquo;").replace("''", "&rdquo;")
    s = s.replace("\\%", "%").replace("\\&", "&amp;").replace("\\_", "_")
    s = s.replace("~", "&nbsp;")
    s = re.sub(r"\\[a-zA-Z]+", "", s)          # drop any leftover commands
    s = s.replace("{", "").replace("}", "")
    return s


def tidy_math(m):
    reps = {r"\times": "&times;", r"\pm": "&plusmn;", r"\leq": "&le;",
            r"\geq": "&ge;", r"\cdot": "&middot;", r"\alpha": "&alpha;",
            r"\beta_2": "&beta;<sub>2</sub>", r"\lambda": "&lambda;",
            r"\omega": "&omega;", r"\Delta\nu": "&Delta;&nu;",
            r"\Delta f": "&Delta;f", r"\mathrm": "", r"\,": " ",
            r"\emph": "", r"\sqrt": "&radic;"}
    for k, v in reps.items():
        m = m.replace(k, v)
    m = re.sub(r"\^\{?([^{}\s]+)\}?", r"<sup>\1</sup>", m)
    m = re.sub(r"_\{?([^{}\s]+)\}?", r"<sub>\1</sub>", m)
    m = re.sub(r"\\[a-zA-Z]+", "", m)
    return m.replace("{", "").replace("}", "")


# ---------------------------------------------------------------------------
def convert(tex):
    out = []
    # ---- title / author ----
    title = re.search(r"\\title\{(.*?)\}\s*\n\s*\\author", tex, re.S)
    if title:
        t = title.group(1).replace("\\\\", " ").strip()
        out.append("<h1>" + inline(t) + "</h1>")

    # ---- abstract ----
    abst = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if abst:
        out.append('<div class="abstract"><h2>Abstract</h2><p>'
                   + inline(" ".join(abst.group(1).split())) + "</p></div>")

    kw = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", tex, re.S)
    if kw:
        out.append('<p class="kw"><b>Index terms &mdash;</b> '
                   + inline(" ".join(kw.group(1).split())) + "</p>")

    # ---- body ----
    body = tex.split(r"\end{IEEEkeywords}", 1)[-1]
    body = body.split(r"\begin{thebibliography}")[0]

    # figures
    def fig_repl(m):
        block = m.group(0)
        src = re.search(r"includegraphics\[[^\]]*\]\{([^}]+)\}", block)
        cap = re.search(r"\\caption\{(.*?)\}\s*\n?\s*\\label", block, re.S)
        if not cap:
            cap = re.search(r"\\caption\{(.*?)\}", block, re.S)
        if not src:
            return ""
        path = os.path.join(config.FIGURES_DIR, src.group(1))
        if not os.path.exists(path):
            return '<p class="missing">[missing figure: {}]</p>'.format(
                src.group(1))
        capt = inline(" ".join(cap.group(1).split())) if cap else ""
        return ('<figure><img src="{}" alt="{}"><figcaption>{}</figcaption>'
                "</figure>".format(embed(path), html.escape(src.group(1)), capt))

    body = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", fig_repl, body, flags=re.S)

    # tables
    def tab_repl(m):
        block = m.group(0)
        cap = re.search(r"\\caption\{(.*?)\}\s*\n?\s*\\label", block, re.S)
        capt = inline(" ".join(cap.group(1).split())) if cap else ""
        return ('<div class="tab"><p class="tabcap">' + capt + "</p>"
                + convert_tabular(block) + "</div>")

    body = re.sub(r"\\begin\{table\}.*?\\end\{table\}", tab_repl, body, flags=re.S)

    # equations -> centred italic line
    body = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}",
                  lambda m: '<p class="eq"><i>' + tidy_math(
                      " ".join(m.group(1).split())) + "</i></p>",
                  body, flags=re.S)

    # lists -- group(1) is the environment name, group(2) is its body
    body = re.sub(r"\\begin\{(itemize|enumerate)\}(.*?)\\end\{\1\}",
                  lambda m: ("<ul>" if m.group(1) == "itemize" else "<ol>")
                  + "".join("<li>" + inline(" ".join(i.split())) + "</li>"
                            for i in re.split(r"\\item", m.group(2))[1:])
                  + ("</ul>" if m.group(1) == "itemize" else "</ol>"),
                  body, flags=re.S)

    # sections then paragraphs
    chunks = re.split(r"(\\section\*?\{[^}]*\}|\\subsection\*?\{[^}]*\})", body)
    for ch in chunks:
        if ch.startswith("\\section"):
            out.append("<h2>" + inline(re.search(r"\{(.*)\}", ch).group(1))
                       + "</h2>")
        elif ch.startswith("\\subsection"):
            out.append("<h3>" + inline(re.search(r"\{(.*)\}", ch).group(1))
                       + "</h3>")
        else:
            for para in re.split(r"\n\s*\n", ch):
                p = para.strip()
                if not p or p.startswith("%"):
                    continue
                if p.startswith("<"):
                    out.append(p)
                    continue
                p = re.sub(r"^%.*$", "", p, flags=re.M).strip()
                if p:
                    out.append("<p>" + inline(" ".join(p.split())) + "</p>")

    # ---- references ----
    refs = re.search(r"\\begin\{thebibliography\}.*?\n(.*?)\\end\{thebibliography\}",
                     tex, re.S)
    if refs:
        out.append("<h2>References</h2><ol class='refs'>")
        for item in re.split(r"\\bibitem\{[^}]*\}", refs.group(1))[1:]:
            out.append("<li>" + inline(" ".join(item.split())) + "</li>")
        out.append("</ol>")
    return "\n".join(out)


CSS = """
:root{--bg:#fbfbfc;--fg:#14161c;--mut:#5b6270;--rule:#dde1e8;--acc:#12615c;
      --card:#fff;--warn:#8a4b00;--warnbg:#fdf3e4}
@media(prefers-color-scheme:dark){:root{--bg:#0e1015;--fg:#e8eaf0;--mut:#a2a9b8;
      --rule:#272c38;--acc:#48c9bd;--card:#161a22;--warn:#e0a253;--warnbg:#2a2010}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;
     font:16px/1.62 Georgia,"Times New Roman",serif;}
.page{max-width:44rem;margin:0 auto;padding:2.5rem 1.25rem 5rem}
h1{font-size:1.95rem;line-height:1.22;margin:0 0 1.6rem;text-wrap:balance}
h2{font-size:1.25rem;margin:2.4rem 0 .7rem;padding-bottom:.3rem;
   border-bottom:1px solid var(--rule);font-variant:small-caps;letter-spacing:.02em}
h3{font-size:1.05rem;margin:1.7rem 0 .5rem;color:var(--acc)}
p{margin:0 0 .95rem;text-align:justify;hyphens:auto}
.abstract{background:var(--card);border:1px solid var(--rule);border-radius:3px;
          padding:1rem 1.2rem;margin-bottom:1.2rem}
.abstract h2{margin:0 0 .5rem;border:0;font-size:1rem}
.abstract p{font-size:.94rem;margin:0}
.kw{font-size:.9rem;color:var(--mut)}
figure{margin:1.8rem 0}
figure img{width:100%;height:auto;border:1px solid var(--rule);border-radius:3px;
           background:#fff}
figcaption{font-size:.85rem;color:var(--mut);margin-top:.5rem;line-height:1.45}
table{border-collapse:collapse;width:100%;font:14px/1.5 system-ui,sans-serif;
      margin:.4rem 0}
th,td{border-bottom:1px solid var(--rule);padding:.45rem .6rem;text-align:left}
th{font-weight:600;border-bottom:1.5px solid var(--mut)}
.tab{margin:1.6rem 0;overflow-x:auto}
.tabcap{font-size:.85rem;color:var(--mut);margin:0 0 .35rem}
.eq{text-align:center;font-size:1.05rem;margin:1.1rem 0}
code{font:13px ui-monospace,monospace;background:var(--rule);padding:.05em .3em;
     border-radius:2px}
ul,ol{margin:0 0 1rem;padding-left:1.4rem}
li{margin-bottom:.4rem}
.refs{font-size:.9rem}
.note{background:var(--warnbg);border-left:3px solid var(--warn);
      padding:.85rem 1.1rem;margin:0 0 2rem;font:14px/1.55 system-ui,sans-serif;
      color:var(--fg)}
.note b{color:var(--warn)}
.missing{color:#c00;font-family:system-ui,sans-serif}
"""


PRINT_CSS = """
@page{size:A4;margin:18mm 16mm}
@media print{
  :root{--bg:#fff;--fg:#111;--mut:#444;--rule:#ccc;--acc:#0a4d49;--card:#fff}
  body{background:#fff;color:#111;font-size:10.5pt}
  .page{max-width:none;padding:0}
  .note{display:none}
  h1{font-size:19pt}
  h2{font-size:12.5pt;page-break-after:avoid}
  h3{font-size:11pt;page-break-after:avoid}
  figure,table,.tab{page-break-inside:avoid}
  figure img{max-height:78mm;object-fit:contain;border:1px solid #ccc}
  figcaption{font-size:8.5pt}
  p{orphans:3;widows:3}
  .abstract{border:1px solid #bbb;background:#fafafa}
  a{color:inherit;text-decoration:none}
}
"""


def main():
    macros = load_macros()
    tex = open(PAPER, encoding="utf-8").read()
    tex = re.sub(r"(?m)^\s*%.*$", "", tex)          # strip comment lines
    tex = substitute_macros(tex, macros)

    leftover = sorted(set(re.findall(r"\\([A-Z]\w+)", tex))
                      - {"IEEEauthorblockN", "IEEEauthorblockA", "IEEEkeywords",
                         "IEEEoverridecommandlockouts"})
    body = convert(tex)

    note = ('<div class="note"><b>Preview only.</b> This is a readable '
            'rendering of <code>paper.tex</code> with all '
            '{} result macros substituted, produced because LaTeX is not '
            'installed here. It does <b>not</b> reproduce IEEE two-column '
            'layout, figure placement or citation numbering &mdash; compile '
            'the real PDF on Overleaf or with MiKTeX. See '
            '<code>paper/README.md</code>.</div>'.format(len(macros)))

    doc = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
           "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
           "<title>Representation Over Architecture</title><style>"
           + CSS + PRINT_CSS + "</style></head><body>"
           "<div class=\"page\">" + note + body + "</div></body></html>")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)

    print("wrote " + OUT)
    print("  macros substituted : {}".format(len(macros)))
    print("  size               : {:.0f} KB".format(len(doc.encode()) / 1024))
    if leftover:
        print("  NOTE: unresolved capitalised commands: "
              + ", ".join(leftover[:10]))
    else:
        print("  no unresolved macros remain")


if __name__ == "__main__":
    main()
