# Figure scripts

These scripts redraw Figs. 1–6 of the paper from `results/results.json` and `data/`.
No DFT code is needed, only `matplotlib` in addition to the main requirements.

```bash
cd scripts/figures && for i in 1 2 3 4 5 6; do python fig$i.py; done
```

The figures are written to `figures/` at the repository root, as `fig1.png` … `fig6.png` (600 dpi)
and matching PDFs. `style.py` holds the shared fonts, sizes and colors, and `icons.py` the drawn
icons used in Figs. 1 and 3. The figures use Arial; without it matplotlib falls back to another
sans-serif font.
