"""
Monte Carlo + closed-form derivation for the Pyrknot planetary parade
(Jane Street, March 2026).

Setup
-----
Pyrknot is a unit sphere (radius R).  Six planets sit at independent uniform
directions v_1, …, v_6 ∈ S² and the star at an independent uniform s ∈ S².
A friend stands at a fixed point  ẑ = (0, 0, 1)  on the surface.

Events:
    A   friend sees all six planets at night
          =  { v_i · ẑ > 0  for every i }  and  { s · ẑ < 0 }
    E   the parade is visible somewhere on Pyrknot at night
          =  { ∃ w ∈ S² :  v_i · w > 0 ∀ i  and  s · w < 0 }
          ⇔   the 7 directions (v_1, …, v_6, −s) all sit in some open
                hemisphere of S².

The tower refinement (small angular horizon shift δ = r/R):
    A_δ  friend's tower of horizon-angle δ sees all six at night
          =  { v_i · ẑ > −sin δ ∀ i }  and  { s · ẑ < −sin δ }.

Exact answer (Euler's formula on 7 great circles)
-------------------------------------------------
Each v_i contributes a "horizon" great circle  { w : v_i · w = 0 }  on the
surface and s contributes the day/night terminator  { w : s · w = 0 }.  Seven
great circles in general position form an arrangement with

    V = 2 · C(7, 2) = 42         (each pair intersects in 2 antipodal points)
    E = 7 · 12     = 84          (each circle is split into 12 arcs)
    F = 2 − V + E  = 44          (Euler,  V − E + F = 2  on S²)

— 44 spherical cells, one per realised sign pattern on the 7 normals.  The
parade-at-night cell is the single pattern (+,+,+,+,+,+,−).

For the friend at the fixed point  ẑ,  independence and uniformity give

    P(A)  =  (1/2)^7  =  1/128.

Wendel's theorem (d=3, n=7 i.i.d. central) says

    P(E)  =  2^{-6} Σ_{k=0}^{2} C(6, k)  =  (1 + 6 + 15)/64  =  22/64  =  11/32
         =  44 / 128,

and therefore

    α  :=  P(A | E)  =  (1/128) / (44/128)  =  1 / 44.

Linear-in-δ expansion for the tower gives β:

    P(A_δ)  =  ((1 + sin δ) / 2)^6 · ((1 − sin δ) / 2)
           =  (1/128) · (1 + sin δ)^6 · (1 − sin δ)
           =  (1/128) · (1 + 5 sin δ + 9 sin² δ + O(sin³ δ))
    α(δ)   =  P(A_δ | E)  =  (1/44) · (1 + 5 sin δ + O(sin² δ))
           =  1/44  +  (5 / 44) δ  +  O(δ²)
⇒   β  =  5 / 44.

Run
---
    python pyrknot_probability.py --batches 10 --batch-size 50000 --delta 0.01
    python pyrknot_probability.py --randomize --batches 5 --batch-size 100000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from fractions import Fraction

import numpy as np

try:
    from scipy.spatial import Delaunay
except ImportError as e:
    raise ImportError(
        "pyrknot_probability requires scipy (pip install scipy)"
    ) from e


N0 = np.array([0.0, 0.0, 1.0])


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n


def random_directions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Shape ``(n, 3)`` unit vectors uniform on S² (Gaussian normalisation)."""
    raw = rng.normal(size=(n, 3))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def random_star_direction(
    rng: np.random.Generator, *, friend_at_night: bool = False
) -> np.ndarray:
    """Direction from Pyrknot's centre toward its star.

    By default ``friend_at_night=False`` and s is uniform on all of S² —
    the sampling regime that matches the official derivation (α = 1/44).

    If ``friend_at_night=True`` the star is rejection-sampled from the
    half-sphere  { s : s · ẑ < 0 },  so the fixed observer at ẑ is guaranteed
    to be on the night side.  Under this extra conditioning the Monte Carlo
    ratio  #A / #E  shifts to the subtly different  1/22  instead of  1/44,
    and is retained only as a sanity check against the earlier model.  The
    Manim scene passes ``True`` purely so the star is always drawn below the
    zenith's horizon in the picture.
    """
    if not friend_at_night:
        return random_directions(1, rng)[0]
    while True:
        s = random_directions(1, rng)[0]
        if np.dot(s, N0) < 0:
            return s


def origin_in_convex_hull(pts: np.ndarray) -> bool:
    """Whether 0 ∈ conv{pts} in R³ (equivalently: pts fit in no open hemisphere)."""
    assert pts.ndim == 2 and pts.shape[1] == 3
    if pts.shape[0] < 4:
        return False
    try:
        tri = Delaunay(pts)
    except Exception:
        return False
    return tri.find_simplex(np.zeros(3)) >= 0


def in_some_open_hemisphere(pts: np.ndarray) -> bool:
    return not origin_in_convex_hull(pts)


def parade_visible_at_night(planets: np.ndarray, star: np.ndarray) -> bool:
    """Event E: the 7 directions (v_1, …, v_6, −s) sit in some open hemisphere."""
    seven = np.vstack([planets, (-unit(star)).reshape(1, 3)])
    return in_some_open_hemisphere(seven)


def friend_sees_parade(
    planets: np.ndarray,
    star: np.ndarray,
    delta: float = 0.0,
    n: np.ndarray | None = None,
) -> bool:
    """Observer at n (default ẑ) with horizon-angle δ sees all six planets *at night*.

    At δ = 0 this is the fixed-ground event A; for δ > 0 it is the tower
    event A_δ (both planet bands widen by sin δ, the night band narrows by
    the same amount — they share one threshold −sin δ).
    """
    if n is None:
        n = N0
    threshold = -math.sin(delta)
    planets_ok = bool(np.all(planets @ n > threshold))
    night_ok = bool(star @ n < threshold)
    return planets_ok and night_ok


def nearest_fraction(x: float, max_den: int = 1_000) -> tuple[Fraction, float]:
    f = Fraction(x).limit_denominator(max_den)
    return f, abs(float(f) - x)


def format_fraction_guess(x: float, max_den: int = 1_000) -> str:
    f, err = nearest_fraction(x, max_den=max_den)
    return f"{f.numerator}/{f.denominator}  (float {float(f):.8f}, err {err:.2e})"


def get_rng_from_env() -> np.random.Generator:
    """For Manim: ``PYRKNOT_RANDOMIZE=1`` for fresh randomness; else ``PYRKNOT_SEED``
    or the default seed 2026."""
    if os.environ.get("PYRKNOT_RANDOMIZE", "").lower() in ("1", "true", "yes"):
        return np.random.default_rng()
    seed_env = os.environ.get("PYRKNOT_SEED")
    if seed_env is not None:
        return np.random.default_rng(int(seed_env))
    return np.random.default_rng(2026)


def run_batches(
    batch_size: int,
    num_batches: int,
    delta: float,
    rng: np.random.Generator,
    *,
    verbose: bool = True,
) -> dict:
    """Sample configurations uniformly and count E, A, A_δ."""
    count_E = count_A = count_B = trials = 0

    for b in range(num_batches):
        e_batch = a_batch = b_batch = 0
        for _ in range(batch_size):
            pts = random_directions(6, rng)
            star = random_star_direction(rng)
            trials += 1

            e = parade_visible_at_night(pts, star)
            a = friend_sees_parade(pts, star, delta=0.0)
            bt = friend_sees_parade(pts, star, delta=delta)

            if e:
                count_E += 1
                e_batch += 1
            if a:
                count_A += 1
                a_batch += 1
            if bt:
                count_B += 1
                b_batch += 1

        if verbose:
            print(
                f"batch {b + 1:>3}/{num_batches}  "
                f"E={e_batch:>6}/{batch_size}  "
                f"A={a_batch:>5}/{batch_size}  "
                f"A_δ={b_batch:>5}/{batch_size}",
                flush=True,
            )

    alpha_hat = count_A / count_E if count_E else float("nan")
    p_B_given_E = count_B / count_E if count_E else float("nan")
    beta_hat = (p_B_given_E - alpha_hat) / delta if delta and count_E else float("nan")

    return {
        "trials": trials,
        "count_E": count_E,
        "count_A": count_A,
        "count_B": count_B,
        "alpha_hat": alpha_hat,
        "p_B_given_E": p_B_given_E,
        "beta_hat": beta_hat,
        "delta": delta,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Monte Carlo for the Pyrknot parade")
    p.add_argument("--batch-size", type=int, default=10_000, help="trials per batch")
    p.add_argument("--batches", type=int, default=10, help="number of batches")
    p.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="angular horizon shift in radians; β is the coefficient of δ in α(δ)",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--randomize", action="store_true", help="nondeterministic RNG")
    p.add_argument("--max-den", type=int, default=1_000, help="max denom for rational guess")
    args = p.parse_args(argv)

    if args.randomize:
        rng = np.random.default_rng()
    elif args.seed is not None:
        rng = np.random.default_rng(args.seed)
    else:
        rng = get_rng_from_env()

    print(
        f"Monte Carlo: {args.batches} batches × {args.batch_size} "
        f"= {args.batches * args.batch_size} trials, δ={args.delta}",
        flush=True,
    )

    stats = run_batches(args.batch_size, args.batches, args.delta, rng)

    print("\n--- totals ---", flush=True)
    print(f"trials: {stats['trials']}", flush=True)
    print(f"count E   : {stats['count_E']}", flush=True)
    print(f"count A   : {stats['count_A']}", flush=True)
    print(f"count A_δ : {stats['count_B']}", flush=True)

    ah = stats["alpha_hat"]
    print(f"\nα̂ = P(A | E) ≈ {ah:.8f}", flush=True)
    if not math.isnan(ah):
        print(
            f"    nearest fraction (den ≤ {args.max_den}): "
            f"{format_fraction_guess(ah, args.max_den)}",
            flush=True,
        )
        print(f"    exact  α = 1/44 = {1 / 44:.8f}", flush=True)

    pbe = stats["p_B_given_E"]
    bh = stats["beta_hat"]
    print(f"\nP(A_δ | E) ≈ {pbe:.8f}   (δ = {args.delta})", flush=True)
    print(f"β̂ = (P(A_δ | E) − α̂) / δ ≈ {bh:.8f}", flush=True)
    if not math.isnan(bh):
        print(
            f"    nearest fraction (den ≤ {args.max_den}): "
            f"{format_fraction_guess(bh, args.max_den)}",
            flush=True,
        )
        print(f"    exact  β = 5/44 = {5 / 44:.8f}", flush=True)
        print(
            f"    |β̂ − 5/44| = {abs(bh - 5 / 44):.6f}"
            "   (reduce --delta for a tighter linearisation)",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
