"""
Planetary Parade (Pyrknot) — ManimGL visualization.

Math summary (see scene labels):
  • Six random sky directions (uniform on S²). Some hemisphere contains all six
    iff the origin is not in their convex hull (Wendel, d=3): P = 2^{-5} Σ_{k=0}^2 C(5,k) = 1/2.
  • Friend at a fixed spot sees all six iff all lie in their celestial hemisphere: (1/2)^6.
  • α = (1/64) / (1/2) = 1/32.
  • Tower (small r/R = δ): planet i visible from top iff colatitude θ_i < π/2 + δ from base;
    P(one) = (1 + sin δ)/2 ≈ 1/2 + δ/2. To first order, P(B_δ ∩ E^c) = o(δ), so
    P(parade from tower | E) ≈ 2 · ((1+sin δ)/2)^6 ⇒ β = 3/16.

Run (after `source .venv/bin/activate`):
  manimgl planetary_parade.py PlanetaryParadeScene -w
  manimgl planetary_parade.py PlanetaryParadeScene -s   # last frame only

Random sky (Manim): set `PYRKNOT_RANDOMIZE=1` or `PYRKNOT_SEED=123`.

Monte Carlo (batched, rational estimates):
  python pyrknot_probability.py --batches 10 --batch-size 20000 --delta 0.05
  python pyrknot_probability.py --randomize --batches 5 --batch-size 100000
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pyrknot_probability import get_rng_from_env
from pyrknot_probability import random_directions

from manimlib import *


class PlanetaryParadeScene(ThreeDScene):
    """3D sketch: Pyrknot, observer, six planets on the sky, visible caps, tower limit."""

    def construct(self):
        R = 1.4
        sky_r = 2.6
        rng = get_rng_from_env()
        dirs = random_directions(6, rng)
        planet_colors = [RED_E, ORANGE, GREEN_E, TEAL, BLUE_E, PURPLE_E]

        # --- Fixed HUD ---
        title = Text("Planetary Parade on Pyrknot", font_size=34)
        title.to_edge(UP, buff=0.25).fix_in_frame()

        subtitle = Text(
            "Six planets · random sky directions · hemisphere visibility",
            font_size=22,
        )
        subtitle.next_to(title, DOWN, buff=0.15).fix_in_frame()

        self.add(title, subtitle)

        # --- Planet Pyrknot ---
        world = Sphere(
            radius=R,
            resolution=(41, 21),
        )
        world.set_color(GREY_E)
        world.set_opacity(0.35)

        # Observer at north pole (+Z); "up" is OUT
        observer = Sphere(radius=0.07, resolution=(13, 7))
        observer.move_to(R * OUT)
        observer.set_color(YELLOW_E)

        # --- Sky shell (wireframe) ---
        sky = Sphere(
            radius=sky_r,
            resolution=(33, 17),
        )
        sky.set_color(GREY_D)
        sky.set_opacity(0.12)

        # Northern celestial hemisphere (from observer): in this Sphere parametrization,
        # v ∈ [π/2, π] is z ≥ 0.
        def visible_shell(delta: float) -> Sphere:
            # θ < π/2 + δ  ⟺  v > π/2 − δ in our (θ = π − v) identification
            v_min = max(1e-3, PI / 2 - delta)
            shell = Sphere(
                radius=sky_r * 0.96,
                u_range=(0, TAU),
                v_range=(v_min, PI),
                resolution=(29, max(4, int(12 + 40 * delta / PI))),
            )
            shell.set_color(BLUE_E)
            shell.set_opacity(0.22)
            return shell

        delta_tracker = ValueTracker(0.0)
        visible_region = visible_shell(0.0)

        def region_updater(m: Mobject):
            d = float(delta_tracker.get_value())
            new_shell = visible_shell(d)
            m.become(new_shell)

        visible_region.add_updater(region_updater)

        # Planet markers on the sky
        planet_mobs = Group()
        for i in range(6):
            p = Sphere(radius=0.11, resolution=(11, 6))
            p.move_to(sky_r * dirs[i])
            p.set_color(planet_colors[i])
            planet_mobs.add(p)

        # Zenith and horizon reference
        zenith_arrow = Arrow(
            R * OUT * 1.05,
            (R + 0.45) * OUT,
            buff=0,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.2,
        )
        zenith_arrow.set_color(YELLOW_B)
        zenith_label = Text("Zenith", font_size=20)
        zenith_label.next_to(zenith_arrow.get_end(), UP, buff=0.08)

        # --- Tower (schematic): base on surface, thin column ---
        tower_h = 0.55
        tower = Cylinder(
            height=tower_h,
            radius=0.06,
            resolution=(12, 8),
        )
        tower.move_to((R + tower_h / 2) * OUT)
        tower.set_color(GREY_C)

        # --- Intro animation ---
        self.set_floor_plane("xy")
        self.play(
            FadeIn(world),
            FadeIn(sky),
            FadeIn(observer),
            FadeIn(zenith_arrow),
            Write(zenith_label),
        )
        self.play(FadeIn(planet_mobs, lag_ratio=0.15))
        self.play(FadeIn(visible_region))
        self.wait(0.5)

        self.play(
            delta_tracker.animate.set_value(0.35),
            self.frame.animate.increment_theta(35 * DEG),
            run_time=3.0,
        )
        self.wait(0.5)

        self.play(FadeIn(tower))
        self.wait(0.3)

        # --- Answer panel ---
        panel = Text(
            "α = 1/32      β = 3/16",
            font_size=40,
            weight=BOLD,
        )
        panel.to_edge(DOWN, buff=0.35).fix_in_frame()

        note = Text(
            "Wendel (d=3, n=6): P(some hemisphere) = 1/2  →  α = (1/2)^6 / (1/2)",
            font_size=18,
        )
        note.next_to(panel, UP, buff=0.2).fix_in_frame()

        self.play(FadeIn(note), FadeIn(panel))
        self.wait(2.0)
