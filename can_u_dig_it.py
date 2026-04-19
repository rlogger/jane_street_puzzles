"""
Jane Street puzzle — "Can U Dig It?" (April 2026).

Puzzle: a 14 × 14 letter grid with a single non-letter (a hyphen) in the last
row.  The official page carries no instructions ("we've drawn a blank with
this puzzle's instructions…") and only promises the answer is a positive
integer.

Decoding:
  1. Title pun.  "Can U Dig It?"  →  "DIG IT"  →  DIGIT.  The answer is a
     number — and "Can U" hints at a U that has been "dug" somewhere it
     doesn't belong.
  2. The only row with a non-letter is the bottom row, which reads
         TUNENTY-TESSIX.
     Read it as TWENTY-SIX with letters buried inside:
         TWENTY → T·UN·ENTY    (the W has been replaced by "UN";
                                the extra U is the U that has "dug" in)
         SIX    → TES·SIX      (TES has been buried in front of SIX)
     The hyphen stays as a bright flashing pointer at the phrase.
  3. Confirmation pass: the number words ONE, SIX, and TEN all sit hidden
     inside the rest of the grid (plus thematic words like ALUMINUM, TIN,
     FIND, ART, …), so "look for numbers" is the intended game.

Answer: 26.

Run:
    python can_u_dig_it.py
"""

from __future__ import annotations

from dataclasses import dataclass


GRID = [
    "rsdifindthsart",
    "ehresodaeetgna",
    "netrhalxhgowip",
    "egedauyueaenrp",
    "ptnnmllmxidnee",
    "ohuinkthanacsm",
    "alnpfyldebsttn",
    "uumjarebemehrw",
    "mithdceigiugts",
    "tlamibftoteget",
    "sailniitniapen",
    "nstoagrniiobrt",
    "ietiryeesprayw",
    "tunenty-tessix",
]
ROWS = len(GRID)
COLS = len(GRID[0])
assert ROWS == 14 and all(len(r) == 14 for r in GRID), "expected a 14 × 14 grid"

DIRS = {
    "N":  (-1,  0),
    "NE": (-1,  1),
    "E":  ( 0,  1),
    "SE": ( 1,  1),
    "S":  ( 1,  0),
    "SW": ( 1, -1),
    "W":  ( 0, -1),
    "NW": (-1, -1),
}


@dataclass(frozen=True)
class Hit:
    word: str
    row: int
    col: int
    direction: str


def find_word(word: str) -> list[Hit]:
    """All occurrences of `word` in the grid (case-insensitive, 8 directions)."""
    w = word.lower()
    L = len(w)
    out: list[Hit] = []
    for r in range(ROWS):
        for c in range(COLS):
            for name, (dr, dc) in DIRS.items():
                er = r + (L - 1) * dr
                ec = c + (L - 1) * dc
                if not (0 <= er < ROWS and 0 <= ec < COLS):
                    continue
                if all(GRID[r + k * dr][c + k * dc] == w[k] for k in range(L)):
                    out.append(Hit(word.upper(), r + 1, c + 1, name))
    return out


def decode_bottom_row() -> tuple[str, str]:
    """Return ``(observed, decoded)`` for the hyphenated last row."""
    bottom = GRID[-1]
    assert "-" in bottom and all("-" not in r for r in GRID[:-1]), (
        "bottom row should be the only row containing a hyphen"
    )
    left, right = bottom.split("-")     # "tunenty", "tessix"
    twenty = left[0] + "W" + left[3:]   # drop the dug-in "UN", restore W
    six = right[-3:]                    # discard the buried "TES"
    return bottom, f"{twenty}-{six}".upper()


def _fmt(hits: list[Hit]) -> str:
    return ", ".join(f"({h.row:2d},{h.col:2d},{h.direction})" for h in hits)


def main() -> None:
    print("=" * 64)
    print("Jane Street  —  Can U Dig It?  (April 2026)")
    print("=" * 64)

    print("\nGrid (14 × 14):")
    for row in GRID:
        print("  " + " ".join(row))

    print("\n-- step 1 : title pun ---------------------------------------")
    print('  "Can U Dig It?"  →  DIG IT  →  DIGIT   ⇒   the answer is a number')
    print('  "Can U …"        →  a U has been dug in somewhere it should not be')

    print("\n-- step 2 : the only non-letter in the grid -----------------")
    bottom, decoded = decode_bottom_row()
    print(f'  bottom row : "{bottom}"   (the only row containing "-")')
    print(f"  decoded    :  {decoded}   (letters dug in to disguise it)")
    print("      TWENTY  →  T · UN · ENTY    W has been replaced by UN")
    print("      SIX     →  TES · SIX        TES buried in front of SIX")

    print("\n-- step 3 : number-word confirmations ------------------------")
    for word in (
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
    ):
        hits = find_word(word)
        if hits:
            print(f"  {word.upper():<6} ×{len(hits):<2}  {_fmt(hits)}")

    print("\n-- step 4 : thematic non-number words ------------------------")
    for word in ("aluminum", "tin", "find", "art", "dig", "trash", "spray"):
        hits = find_word(word)
        if hits:
            print(f"  {word.upper():<9} ×{len(hits):<2}  {_fmt(hits)}")

    print("\n" + "=" * 64)
    print("  Answer: 26")
    print("=" * 64)


if __name__ == "__main__":
    main()
