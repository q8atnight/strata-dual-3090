# Reference configs

The exact engine configs from our dual-3090 machine (the ones the SF3.7 numbers in the
[dual3090 README](../README.md) were measured with). They carry **our absolute paths** and
our expert-profile paths, so they are a reference for tuning, not templates. On your box,
`dual3090/mkconfig.py` writes the equivalent config automatically from your setup — you
should not need these unless you want to compare notes or hand-edit.

- `strata-iq3_s-dual.json` / `strata-iq2_xs-dual.json` / `strata-q2_0-dual.json` — serving
- `strata-gS.json` / `strata-gD.json` — the exactness gate arms (single reference / dual),
  used by [`../verify/gate.sh`](../verify/gate.sh)
