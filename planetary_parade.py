"""
Planetary Parade (Pyrknot) — pedagogical ManimGL walkthrough.

The scene builds the Jane Street derivation one layer at a time:

  1.  Pyrknot and the friend at ẑ.
  2.  A single planet v ⇒ one great circle  { w : v · w = 0 } ; the visible
      cap is the closed hemisphere on v's side.
  3.  The star s ⇒ a seventh great circle — the day/night terminator — and
      a night hemisphere  { w : s · w < 0 }.
  4.  All six planets + the star ⇒ seven great circles on S².
  5.  Euler on the sphere:   V − E + F = 2
           V = 2·C(7, 2) = 42      E = 7 · 12 = 84       F = 44
  6.  Exactly one of the 44 cells has sign pattern (+, +, +, +, +, +, −):
      the parade-at-night region.  P(A) = (1/2)⁷ = 1/128,  P(E) = 11/32 =
      44/128,  so  α = P(A | E) = 1/44.
  7.  Tower of horizon angle δ = r/R widens every planet band by δ (six
      outward arrows on the sphere) while the night band narrows by the
      same δ (one inward arrow).  Net  6 − 1 = 5,  so β = 5/44.

Run
----
    manimgl planetary_parade.py PlanetaryParadeScene -w
    PYRKNOT_RANDOMIZE=1 manimgl planetary_parade.py PlanetaryParadeScene -w

The last frame also works as a still:
    manimgl planetary_parade.py PlanetaryParadeScene -s
"""

from __future__ import annotations

import math
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


# ---------- spherical helpers ----------

_Z_AXIS = np.array([0.0, 0.0, 1.0])


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def _perp_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal ``(e1, e2)`` spanning the plane perpendicular to ``normal``."""
    n = _unit(normal)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(tmp - np.dot(tmp, n) * n)
    e2 = np.cross(n, e1)
    return e1, e2


def great_circle(
    normal: np.ndarray,
    radius: float = 1.0,
    color: str = WHITE,
    stroke_width: float = 3.0,
) -> ParametricCurve:
    """The great circle  { w ∈ S² : normal · w = 0 }  at the given radius."""
    e1, e2 = _perp_basis(normal)
    return ParametricCurve(
        lambda t: radius * (math.cos(t) * e1 + math.sin(t) * e2),
        t_range=[0.0, TAU, TAU / 96],
        color=color,
        stroke_width=stroke_width,
    )


def hemisphere_cap(
    normal: np.ndarray,
    radius: float = 1.0,
    color: str = BLUE_E,
    opacity: float = 0.14,
) -> Sphere:
    """Closed hemisphere  { w : normal · w ≥ 0 }  shaded at the given radius."""
    cap = Sphere(
        radius=radius,
        u_range=(0, TAU),
        v_range=(PI / 2, PI),
        resolution=(33, 17),
    )
    cap.set_color(color)
    cap.set_opacity(opacity)
    n = _unit(normal)
    axis = np.cross(_Z_AXIS, n)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        if n[2] < 0:
            cap.rotate(PI, axis=RIGHT)
    else:
        angle = math.acos(float(np.clip(np.dot(_Z_AXIS, n), -1.0, 1.0)))
        cap.rotate(angle, axis=axis / axis_norm)
    return cap


def radial_arrow(
    direction: np.ndarray,
    start_radius: float,
    length: float = 0.32,
    color: str = WHITE,
) -> Arrow:
    """Short 3D arrow pointing radially outward from ``start_radius·direction``."""
    d = _unit(direction)
    return Arrow(
        d * start_radius,
        d * (start_radius + length),
        buff=0.0,
        stroke_width=5,
        max_tip_length_to_length_ratio=0.30,
    ).set_color(color)


# ---------- main scene ----------

class PlanetaryParadeScene(ThreeDScene):
    """Pedagogical walk-through: 7 great circles → 44 cells → α = 1/44, β = 5/44."""

    def construct(self) -> None:
        R = 1.4
        sky_r = 2.55
        planet_colors = [RED_E, ORANGE, GREEN_E, TEAL, BLUE_E, PURPLE_E]

        rng = get_rng_from_env()
        planets = random_directions(6, rng)
        # Rejection-sample only so the star visibly sits below the friend's
        # horizon; the probability values we report are the unconditioned ones.
        star = random_star_direction(rng, friend_at_night=True)
        anti_star = -star

        # ---- fixed HUD: title + moving caption -----------------------------
        title = Text("Planetary Parade  —  solving for  α  and  β",
                     font_size=34, weight=BOLD)
        title.to_edge(UP, buff=0.25).fix_in_frame()
        self.add(title)

        self._caption: Text | None = None

        def say(text: str, run_time: float = 0.7) -> None:
            new = Text(text, font_size=22)
            new.to_edge(DOWN, buff=0.28).fix_in_frame()
            if self._caption is None:
                self.play(FadeIn(new), run_time=run_time)
            else:
                self.play(FadeOut(self._caption), FadeIn(new), run_time=run_time)
            self._caption = new

        def clear_caption(run_time: float = 0.3) -> None:
            if self._caption is not None:
                self.play(FadeOut(self._caption), run_time=run_time)
                self._caption = None

        # ---- stage 1: Pyrknot and the friend -------------------------------
        self.set_floor_plane("xy")

        world = Sphere(radius=R, resolution=(41, 21))
        world.set_color(GREY_E).set_opacity(0.28)
        friend = Sphere(radius=0.07, resolution=(13, 7))
        friend.move_to(R * OUT).set_color(YELLOW_E)
        zen_arrow = Arrow(
            R * OUT * 1.02,
            (R + 0.45) * OUT,
            buff=0.0,
            stroke_width=4,
            max_tip_length_to_length_ratio=0.22,
        ).set_color(YELLOW_B)
        zen_label = Text("friend  ẑ", font_size=18)
        zen_label.next_to(zen_arrow.get_end(), UP, buff=0.08)

        self.play(
            FadeIn(world),
            FadeIn(friend),
            FadeIn(zen_arrow),
            Write(zen_label),
            run_time=1.3,
        )
        say("Pyrknot (radius R).  The friend stands at the fixed point ẑ.")
        self.wait(0.3)

        # ---- stage 2: one planet, one horizon ------------------------------
        planet_balls = []
        for i in range(6):
            ball = Sphere(radius=0.10, resolution=(11, 6))
            ball.move_to(sky_r * planets[i])
            ball.set_color(planet_colors[i])
            planet_balls.append(ball)

        gc_planets = [
            great_circle(planets[i], R, color=planet_colors[i]) for i in range(6)
        ]
        cap_planets = [
            hemisphere_cap(planets[i], R * 0.995,
                           color=planet_colors[i], opacity=0.14)
            for i in range(6)
        ]

        say("A planet in direction v is visible from the half of Pyrknot where "
            "v · w ≥ 0.  Its horizon is the great circle v · w = 0.")
        self.play(FadeIn(planet_balls[0]), run_time=0.6)
        self.play(
            ShowCreation(gc_planets[0]),
            FadeIn(cap_planets[0]),
            run_time=1.8,
        )
        self.wait(0.4)

        # ---- stage 3: star + terminator + night hemisphere -----------------
        star_ball = Sphere(radius=0.16, resolution=(13, 7))
        star_ball.move_to(sky_r * star).set_color("#ffcc66")
        anti_ball = Sphere(radius=0.08, resolution=(11, 6))
        anti_ball.move_to(sky_r * anti_star).set_color("#4466aa")

        star_lbl = Text("star s", font_size=18)
        star_lbl.next_to(star_ball, OUT * 0.3 + LEFT * 0.2)
        anti_lbl = Text("−s  (night)", font_size=16)
        anti_lbl.next_to(anti_ball, OUT * 0.2 + RIGHT * 0.15)

        terminator = great_circle(star, R, color=GOLD)
        night_cap = hemisphere_cap(anti_star, R * 0.990, color=BLUE_E, opacity=0.14)

        say("Daylight blots out the sky, so the star adds a 7th great circle — "
            "the day / night terminator s · w = 0.")
        self.play(
            FadeIn(star_ball),
            FadeIn(anti_ball),
            FadeIn(star_lbl),
            FadeIn(anti_lbl),
            ShowCreation(terminator),
            FadeIn(night_cap),
            run_time=2.0,
        )
        self.wait(0.4)

        # ---- stage 4: five more planets, seven circles ---------------------
        say("Five more planets bring five more horizons:  seven great circles "
            "in total.")
        self.play(
            *[FadeIn(b) for b in planet_balls[1:]],
            *[ShowCreation(c) for c in gc_planets[1:]],
            *[FadeIn(c) for c in cap_planets[1:]],
            run_time=2.6,
        )
        self.play(self.frame.animate.increment_theta(35 * DEG), run_time=2.0)
        self.wait(0.3)

        # ---- stage 5: Euler on the sphere ----------------------------------
        say("Euler on S²:   V − E + F = 2.   Seven circles in general "
            "position give…")
        euler_lines = VGroup(
            Text("V = 2 · C(7, 2) = 42", font_size=24),
            Text("E = 7 · 12       = 84", font_size=24),
            Text("F = 2 + E − V    = 44", font_size=26, weight=BOLD, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        euler_lines.to_corner(UL, buff=0.4).fix_in_frame()
        self.play(FadeIn(euler_lines), run_time=1.0)
        self.wait(2.0)

        # ---- stage 6: α = 1/44 ---------------------------------------------
        say("Exactly one of the 44 cells has sign pattern (+, +, +, +, +, +, −) — "
            "the parade-at-night region.")
        alpha_lines = VGroup(
            Text("P(A) = (1/2)⁷      = 1/128", font_size=22),
            Text("P(E) = 11/32       = 44/128", font_size=22),
            Text("α = P(A | E) = 1/44", font_size=28, weight=BOLD, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        alpha_lines.to_corner(UR, buff=0.4).fix_in_frame()
        self.play(FadeIn(alpha_lines), run_time=1.0)
        self.wait(2.4)

        # ---- stage 7: tower + β = 5/44 -------------------------------------
        say("A tower of horizon angle δ = r/R widens every planet band by δ…")
        tower_h = 0.55
        tower = Cylinder(height=tower_h, radius=0.055, resolution=(12, 8))
        tower.move_to((R + tower_h / 2) * OUT).set_color(GREY_C)
        self.play(FadeIn(tower), run_time=0.6)

        plus_arrows = VGroup(*[
            radial_arrow(planets[i], R, length=0.32, color=planet_colors[i])
            for i in range(6)
        ])
        self.play(FadeIn(plus_arrows, lag_ratio=0.08), run_time=1.8)

        say("…and narrows the night band by the same δ.")
        minus_arrow = Arrow(
            anti_star * (R + 0.32),
            anti_star * R,
            buff=0.0,
            stroke_width=5,
            max_tip_length_to_length_ratio=0.30,
        ).set_color("#66aaff")
        self.play(FadeIn(minus_arrow), run_time=0.8)

        self.play(FadeOut(euler_lines), run_time=0.4)
        beta_lines = VGroup(
            Text("+ 6 planet bands widen by δ", font_size=22),
            Text("− 1 night band narrows by δ", font_size=22),
            Text("β = 5/44   (net  6 − 1 = 5)", font_size=28,
                 weight=BOLD, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        beta_lines.to_corner(UL, buff=0.4).fix_in_frame()
        self.play(FadeIn(beta_lines), run_time=1.0)
        self.wait(2.6)

        # ---- stage 8: final answer card ------------------------------------
        clear_caption()
        final_panel = Text("α = 1/44        β = 5/44",
                           font_size=48, weight=BOLD)
        final_panel.to_edge(DOWN, buff=0.40).fix_in_frame()
        self.play(FadeIn(final_panel), run_time=0.8)
        self.wait(2.8)
