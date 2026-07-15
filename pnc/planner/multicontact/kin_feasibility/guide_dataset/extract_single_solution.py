"""Extract a single solution from a guide dataset .npz file.

Usage (from repo root):
    python pnc/planner/multicontact/kin_feasibility/guide_dataset/extract_single_solution.py \\
        --input guide_dataset_dyn_multiple_test_rnd10.npz \\
        --solution 1 \\
        --output solution_1.npz

The script verifies that the requested solution has is_optimal=True before
saving.  If the solution is infeasible (is_optimal=False), it reports all
valid solution indices so the user can try again.

Per-solution arrays (first axis = n_solutions) are sliced to a single
timestep, dropping that axis.  Shared arrays (scalars and name arrays) are
copied as-is.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def extract_solution(input_path: str, solution_idx: int, output_path: str) -> None:
    if not os.path.isfile(input_path):
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    data = np.load(input_path, allow_pickle=True)
    keys = list(data.keys())

    if "is_optimal" not in keys:
        print("Error: 'is_optimal' key not found in the dataset.", file=sys.stderr)
        sys.exit(1)

    is_optimal = data["is_optimal"]
    n_solutions = is_optimal.shape[0]

    if solution_idx < 0 or solution_idx >= n_solutions:
        print(
            f"Error: solution index {solution_idx} is out of range. "
            f"The file contains {n_solutions} solutions (indices 0–{n_solutions - 1}).",
            file=sys.stderr,
        )
        sys.exit(1)

    if not bool(is_optimal[solution_idx]):
        valid_indices = list(np.where(is_optimal)[0])
        if valid_indices:
            print(
                f"Solution {solution_idx} is infeasible (is_optimal=False). "
                f"Please try one of the following valid solution indices: {valid_indices}"
            )
        else:
            print(
                f"Solution {solution_idx} is infeasible (is_optimal=False). "
                "No valid solutions exist in this file."
            )
        sys.exit(1)

    # Separate per-solution arrays (shape[0] == n_solutions) from shared ones.
    extracted: dict[str, np.ndarray] = {}
    for key in keys:
        arr = data[key]
        if arr.ndim > 0 and arr.shape[0] == n_solutions:
            extracted[key] = arr[solution_idx]
        else:
            extracted[key] = arr[()]  # preserve scalars and shared arrays

    np.savez(output_path, **extracted)

    print(f"Extracted solution {solution_idx} from '{input_path}' → '{output_path}'")
    print(f"  {len(keys)} arrays saved, {n_solutions} → 1 solution.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a single trajectory solution from a guide dataset .npz file."
    )
    parser.add_argument(
        "--input",
        default="guide_dataset_dyn_multiple_test_rnd10.npz",
        help="Path to the source .npz dataset (default: guide_dataset_dyn_multiple_test_rnd10.npz)",
    )
    parser.add_argument(
        "--solution",
        type=int,
        required=True,
        help="Index of the solution to extract (0-based).",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output .npz file path. "
            "Defaults to '<input_stem>_solution_<N>.npz' in the same directory."
        ),
    )
    args = parser.parse_args()

    output_path = args.output
    if not output_path:
        base, _ = os.path.splitext(args.input)
        output_path = f"{base}_solution_{args.solution}.npz"

    extract_solution(args.input, args.solution, output_path)


if __name__ == "__main__":
    main()
