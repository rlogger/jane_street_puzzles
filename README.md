# jane_street_puzzles

Solutions and supporting code for [Jane Street's monthly puzzles](https://www.janestreet.com/puzzles/) — probability, geometry, word-search, and optional [ManimGL](https://github.com/3b1b/manim) animations.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Can U Dig It? — April 2026

Fourteen-by-fourteen letter grid with no printed instructions ("we've drawn a blank…") and a promise that the answer is a positive integer.

The puzzle is a self-referential word-search:

- **Title pun.** *Can U Dig It?* → *DIG IT* → **DIGIT**. The answer is a number, and a stray *U* has been dug in somewhere.
- **Bottom row.** The only row with a non-letter is `tunenty-tessix`, i.e. **TWENTY-SIX** with letters buried:
  - `TWENTY → T·UN·ENTY` (the *W* replaced by *UN* — the dug-in *U*)
  - `SIX → TES·SIX` (*TES* buried in front of *SIX*)
- **Confirmation.** `ONE`, `SIX`, `TEN` (and thematic `ALUMINUM`, `TIN`, `FIND`, `ART`, `TRASH`, `SPRAY`, …) sit inside the grid as hidden words, so "look for numbers" is the intended game.

**Answer: 26.**

```bash
python can_u_dig_it.py
```

## Planetary Parade (Pyrknot) — March 2026

Probability that a friend at a fixed spot on Pyrknot witnesses all six planets simultaneously at night, plus the first-order correction for building a tower of height `r ≪ R`.

Official answers ([solution page](https://www.janestreet.com/puzzles/planetary-parade-solution)):

- **α = 1/44** — friend on the surface.
- **β = 5/44** — coefficient of `r/R` in α(δ) for a tower of angular horizon shift δ = r/R.

Seven great circles (six planet horizons + the day/night terminator) cut S² into `V − E + F = 42 − 84 + 44 = 44` cells. One of those 44 cells is the parade-at-night region, hence α = 1/44. Widening each planet band by `sin δ` while narrowing the night band by the same amount gives `α(δ) = 1/44 + (5/44)δ + O(δ²)`.

### Monte Carlo

```bash
python pyrknot_probability.py --batches 10 --batch-size 50000 --delta 0.01 --seed 42
python pyrknot_probability.py --randomize --batches 5 --batch-size 100000
```

### Manim

```bash
manimgl planetary_parade.py PlanetaryParadeScene -w
PYRKNOT_RANDOMIZE=1 manimgl planetary_parade.py PlanetaryParadeScene -w
```

Install [FFmpeg](https://ffmpeg.org/) for rendered video. Optional LaTeX only if you use `Tex` elsewhere.

## License

MIT — see [LICENSE](LICENSE).
