# Paper sources

Working draft of "Towards Continual Learning for SmaTables" (spec and current state:
`docs/paper-plan.md`). Venue is still open; the skeleton assumes Springer LNCS
(`llncs.cls` + `splncs04.bst`, both shipped with TeX Live).

Build (output in `paper/build/`, gitignored):

```bash
cd paper && latexmk -pdf -interaction=nonstopmode -output-directory=build main.tex
```

Layout: `main.tex` (front matter, abstract, section inputs), `sections/1..6-*.tex`
(one file per section, outline as comments, open items as red `\todo{}` and blue
`\result{}` markers), `tables/` (generated tables; `memory-feasibility.tex` is produced
from the stage-1 `memory.json` probes), `figures/` (empty until results exist),
`references.bib` (verified entries first, TODO stubs at the end — do not cite a stub
until its fields are filled).

Note: the April plan said to keep the paper private until camera-ready. The GitHub
repository is currently public.
