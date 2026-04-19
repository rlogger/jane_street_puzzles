"""
Planetary Parade (Pyrknot) — ManimGL visualization.

Math summary (see scene labels):
  • Six random planet directions + star direction s (uniform on S²).  The event
    “parade visible somewhere on Pyrknot at night” is (v_1,…,v_6,−s) ∈ some
    open hemisphere; Wendel (n=7, d=3): P(E) = 11/32.
  • Seven great circles (one per v_i plus one for the terminator) tile S² into
    V=42, E=84, F=44 cells (Euler).  Friend at fixed ẑ sits in the parade-at-
    night cell with probability (1/2)^7 = 1/128 = (1/44)·(11/32), so
        α = P(A | E) = 1/44.
  • Tower of horizon angle δ = r/R: widens each planet band by sin δ and
    narrows the night band by the same amount, giving
        α(δ) = 1/44 + (5/44) δ + O(δ²)     ⇒     β = 5/44.

Run (after `source .venv/bin/activate`):
  manimgl planetary_parade.py PlanetaryParadeScene -w
  manimgl planetary_parade.py PlanetaryParadeScene -s   # last frame only

Random sky (Manim): set `PYRKNOT_RANDOMIZE=1` or `PYRKNOT_SEED=123`.

Monte Carlo (batched, rational estimates):
  python pyrknot_probability.py --batches 10 --batch-size 50000 --delta 0.01
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
from pyrknot_probability import random_star_direction

from manimlib import *


class PlanetaryParadeScene(ThreeDScene):
    """3D sketch: Pyrknot, observer, six planets on the sky, visible caps, tower limit."""

    def construct(self):
        R = 1.4
        sky_r = 2.6
        rng = get_rng_from_env()
        dirs = random_directions(6, rng)
        # Friend at zenith is on the night side: star below local horizon (s·ê_z < 0).
        star_dir = random_star_direction(rng, friend_at_night=True)
        anti_dir = -star_dir
        planet_colors = [RED_E, ORANGE, GREEN_E, TEAL, BLUE_E, PURPLE_E]

        # --- Fixed HUD ---
        title = Text("Planetary Parade on Pyrknot", font_size=34)
        title.to_edge(UP, buff=0.25).fix_in_frame()

        subtitle = Text(
            "Six planets + star · anti-star (−s) completes the night-sky hemisphere test",
            font_size=20,
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

        star_ball = Sphere(radius=0.16, resolution=(13, 7))
        star_ball.move_to(sky_r * star_dir)
        star_ball.set_color("#ffcc66")

        anti_ball = Sphere(radius=0.09, resolution=(11, 6))
        anti_ball.move_to(sky_r * anti_dir)
        anti_ball.set_color("#4466aa")

        star_lbl = Text("star s", font_size=18)
        star_lbl.next_to(star_ball, OUT * 0.3 + LEFT * 0.2)
        anti_lbl = Text("−s (night)", font_size=16)
        anti_lbl.next_to(anti_ball, OUT * 0.2 + RIGHT * 0.15)

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
        self.play(
            FadeIn(planet_mobs, lag_ratio=0.15),
            FadeIn(star_ball),
            FadeIn(anti_ball),
            FadeIn(star_lbl),
            FadeIn(anti_lbl),
        )
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
            "α = 1/44      β = 5/44",
            font_size=40,
            weight=BOLD,
        )
        panel.to_edge(DOWN, buff=0.35).fix_in_frame()

        note = Text(
            "7 great circles ⇒ 44 spherical cells;  α = (1/128)/(11/32) = 1/44,  β = 5/44 (linear in δ)",
            font_size=15,
        )
        note.next_to(panel, UP, buff=0.2).fix_in_frame()

        self.play(FadeIn(note), FadeIn(panel))
        self.wait(2.0)
