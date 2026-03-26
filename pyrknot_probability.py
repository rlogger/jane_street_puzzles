"""
Monte Carlo estimation for the Pyrknot planetary parade (6 random sky directions).

  • E: all six directions lie in some open hemisphere  ⇔  0 ∉ conv{v_i}
  • A: friend at north pole sees all six  ⇔  v_i·ê_z > 0 for all i
  • B_δ: tower visibility  ⇔  v_i·ê_z > -sin(δ) for all i  (δ = r/R)

Estimates α ≈ P(A|E) and optionally P(B_δ|E) and β from (P(B_δ|E) - α) / δ for small δ.

Run:
  python pyrknot_probability.py --batches 10 --batch-size 20000
  python pyrknot_probability.py --help
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
    raise ImportError("pyrknot_probability requires scipy (e.g. pip install scipy)") from e


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n


def random_directions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Shape (n, 3) unit vectors, uniform on S² (Gaussian normalization)."""
    raw = rng.normal(size=(n, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def not_in_any_open_hemisphere(pts: np.ndarray) -> bool:
    """
    True iff the six unit vectors are NOT contained in any open hemisphere,
    i.e. iff the origin lies in their convex hull (3D).
    pts: (6, 3)
    """
    assert pts.shape == (6, 3)
    try:
        tri = Delaunay(pts)
    except Exception:
        # Degenerate configuration; treat as in some hemisphere (numerical edge case)
        return False
    return tri.find_simplex(np.zeros(3)) >= 0


def hemisphere_event_E(pts: np.ndarray) -> bool:
    """There exists a surface location from which all six are visible (some hemisphere)."""
    return not not_in_any_open_hemisphere(pts)


def friend_sees_all(pts: np.ndarray, n: np.ndarray | None = None) -> bool:
    """Observer 'up' direction n (default north pole)."""
    if n is None:
        n = np.array([0.0, 0.0, 1.0])
    return bool(np.all(pts @ n > 0))


def tower_sees_all(pts: np.ndarray, delta: float, n: np.ndarray | None = None) -> bool:
    """δ = r/R small; visible iff colatitude < π/2 + δ  ⇔  v·n > cos(π/2+δ) = -sin(δ)."""
    if n is None:
        n = np.array([0.0, 0.0, 1.0])
    thresh = -math.sin(delta)
    return bool(np.all(pts @ n > thresh))


def nearest_fraction(x: float, max_den: int = 10_000) -> tuple[Fraction, float]:
    """Best rational approximation with denominator ≤ max_den; returns (fraction, error)."""
    f = Fraction(x).limit_denominator(max_den)
    err = abs(float(f) - x)
    return f, err


def format_fraction_guess(x: float, max_den: int = 1000) -> str:
    f, err = nearest_fraction(x, max_den=max_den)
    return f"{f.numerator}/{f.denominator}  (float {float(f):.8f}, err {err:.2e})"


def get_rng_from_env() -> np.random.Generator:
    """For Manim: PYRKNOT_RANDOMIZE=1 for fresh randomness; else PYRKNOT_SEED=int or default."""
    env_rand = os.environ.get("PYRKNOT_RANDOMIZE", "").lower() in ("1", "true", "yes")
    if env_rand:
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
    """
    Accumulate counts over batches. Unconditional samples; estimate α by (#A)/(#E) among trials.
    """
    count_E = 0
    count_A = 0
    count_B_and_E = 0

    total_trials = 0

    for b in range(num_batches):
        e_batch = 0
        a_batch = 0
        be_batch = 0

        for _ in range(batch_size):
            pts = random_directions(6, rng)
            total_trials += 1
            e = hemisphere_event_E(pts)
            a = friend_sees_all(pts)
            tower = tower_sees_all(pts, delta)

            if e:
                count_E += 1
                e_batch += 1
            if a:
                count_A += 1
                a_batch += 1
                assert e  # A ⇒ E
            if e and tower:
                count_B_and_E += 1
                be_batch += 1

        if verbose:
            print(
                f"batch {b + 1}/{num_batches}  "
                f"E={e_batch}/{batch_size}  "
                f"A={a_batch}/{batch_size}  "
                f"B∩E={be_batch}/{batch_size}",
                flush=True,
            )

    alpha_hat = count_A / count_E if count_E > 0 else float("nan")
    p_B_given_E = count_B_and_E / count_E if count_E > 0 else float("nan")
    beta_hat = (p_B_given_E - alpha_hat) / delta if delta > 0 and count_E > 0 else float("nan")

    return {
        "trials": total_trials,
        "count_E": count_E,
        "count_A": count_A,
        "count_B_and_E": count_B_and_E,
        "alpha_hat": alpha_hat,
        "p_B_given_E": p_B_given_E,
        "beta_hat": beta_hat,
        "delta": delta,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Monte Carlo for Pyrknot parade probabilities")
    p.add_argument("--batch-size", type=int, default=10_000, help="trials per batch")
    p.add_argument("--batches", type=int, default=10, help="number of batches")
    p.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="r/R for tower (use small values, e.g. 0.005–0.02, for β; "
        "β_hat = (P(B_δ|E)-α_hat)/δ has O(δ) bias when δ is not ≪ 1)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default: env PYRKNOT_SEED or 2026 if unset)",
    )
    p.add_argument(
        "--randomize",
        action="store_true",
        help="ignore seed; use nondeterministic RNG",
    )
    p.add_argument(
        "--max-den",
        type=int,
        default=1000,
        help="max denominator for rational guess",
    )
    args = p.parse_args(argv)

    if args.randomize:
        rng = np.random.default_rng()
    elif args.seed is not None:
        rng = np.random.default_rng(args.seed)
    else:
        rng = get_rng_from_env()

    print(
        f"Monte Carlo: {args.batches} batches × {args.batch_size} = "
        f"{args.batches * args.batch_size} trials, δ={args.delta}",
        flush=True,
    )

    stats = run_batches(
        args.batch_size,
        args.batches,
        args.delta,
        rng,
        verbose=True,
    )

    print("\n--- Totals ---", flush=True)
    print(f"trials:     {stats['trials']}", flush=True)
    print(f"count E:    {stats['count_E']}", flush=True)
    print(f"count A:    {stats['count_A']}", flush=True)
    print(f"count B∩E:  {stats['count_B_and_E']}", flush=True)

    ah = stats["alpha_hat"]
    print(f"\nα_hat = P(A|E) ≈ {ah:.8f}", flush=True)
    if not math.isnan(ah):
        print(f"    nearest fraction (den≤{args.max_den}): {format_fraction_guess(ah, args.max_den)}", flush=True)
        print(f"    exact α = 1/32 = {1/32:.8f}", flush=True)

    pbe = stats["p_B_given_E"]
    bh = stats["beta_hat"]
    print(f"\nP(B_δ|E) ≈ {pbe:.8f}  (δ={args.delta})", flush=True)
    print(f"β_hat = (P(B_δ|E) - α_hat) / δ ≈ {bh:.8f}", flush=True)
    if not math.isnan(bh):
        print(f"    nearest fraction (den≤{args.max_den}): {format_fraction_guess(bh, args.max_den)}", flush=True)
        print(f"    exact β = 3/16 = {3/16:.8f}", flush=True)
        print(
            f"    |β_hat - 3/16| = {abs(bh - 3/16):.6f}  (reduce --delta for a tighter linearization)",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
