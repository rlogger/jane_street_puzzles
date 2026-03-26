# jane_street_puzzles

Jane Street–style puzzles: probability, geometry, and optional [ManimGL](https://github.com/3b1b/manim) animations.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Planetary Parade (Pyrknot)

Monte Carlo estimation and a 3D scene for six random sky directions, hemisphere visibility, and a small-tower limit.

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
