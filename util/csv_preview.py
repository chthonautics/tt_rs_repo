#!/usr/bin/env python3
"""Preview the first rows of a CSV file as a table that fits the terminal.

Wide tables are not allowed to overflow: the columns are split into blocks
that each fit the available width, and the row number is repeated in every
block so that rows stay identifiable across blocks.

Usage:
    python3 csv_preview.py <file.csv> [--rows N] [--width N]
"""

from __future__ import annotations

import argparse
import shutil
import textwrap
from pathlib import Path

import pandas as pd

GAP = "  "
ELLIPSIS = "..."


def _emit(text: str, limit: int) -> None:
    """Print `text`, wrapped to `limit` characters."""
    for line in textwrap.wrap(text, width=limit) or [text]:
        print(line)


def _fmt(value) -> str:
    """Render one cell value as text (blank-ish values become NaN)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "NaN"
    return str(value)


def _trunc(text: str, width: int) -> str:
    """Shorten `text` to `width` characters, marking the cut with an ellipsis."""
    if len(text) <= width:
        return text
    if width <= len(ELLIPSIS):
        return ELLIPSIS[:width]
    return text[: width - len(ELLIPSIS)] + ELLIPSIS


def _pack(widths: list[int], available: int) -> list[list[int]]:
    """Greedily split column indexes into blocks that fit `available` chars."""
    blocks: list[list[int]] = []
    current: list[int] = []
    used = 0

    for index, width in enumerate(widths):
        needed = width + (len(GAP) if current else 0)
        if current and used + needed > available:
            blocks.append(current)
            current, used = [index], width
        else:
            current.append(index)
            used += needed

    if current:
        blocks.append(current)
    return blocks


def render_table(frame: pd.DataFrame, limit: int) -> str:
    """Return `frame` rendered as one or more width-limited blocks."""
    headers = [str(name) for name in frame.columns]
    if not headers:
        return "(no columns)"

    columns = [
        [_fmt(value) for value in frame.iloc[:, position].tolist()]
        for position in range(len(headers))
    ]
    labels = [str(position) for position in range(len(frame))]

    # The row-number column is repeated in front of every block.
    row_number_width = max([len("#")] + [len(label) for label in labels])
    available = max(10, limit - row_number_width - len(GAP))

    natural = [
        max([len(header)] + [len(value) for value in cells])
        for header, cells in zip(headers, columns)
    ]
    # A single over-wide column is clipped instead of breaking the layout.
    widths = [min(width, available) for width in natural]
    numeric = [
        pd.api.types.is_numeric_dtype(frame.dtypes.iloc[position])
        for position in range(len(headers))
    ]

    def align(text: str, width: int, right: bool) -> str:
        return text.rjust(width) if right else text.ljust(width)

    blocks = _pack(widths, available)
    lines: list[str] = []

    for block_number, block in enumerate(blocks):
        if len(blocks) > 1:
            caption = f"[columns {block[0] + 1}-{block[-1] + 1} of {len(headers)}]"
            if len(caption) > limit:
                caption = f"[cols {block[0] + 1}-{block[-1] + 1}/{len(headers)}]"
            lines.extend(textwrap.wrap(caption, width=limit) or [caption])

        header_cells = ["#".rjust(row_number_width)]
        header_cells += [
            align(_trunc(headers[i], widths[i]), widths[i], numeric[i])
            for i in block
        ]
        lines.append(GAP.join(header_cells).rstrip())

        rule = ["-" * row_number_width] + ["-" * widths[i] for i in block]
        lines.append(GAP.join(rule))

        for row, label in enumerate(labels):
            cells = [label.rjust(row_number_width)]
            cells += [
                align(_trunc(columns[i][row], widths[i]), widths[i], numeric[i])
                for i in block
            ]
            lines.append(GAP.join(cells).rstrip())

        if block_number != len(blocks) - 1:
            lines.append("")

    return "\n".join(lines)


def preview(path: Path, rows: int, limit: int) -> None:
    """Load `path` and print its first `rows` rows in a width-limited table."""
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"Error: file not found: {path}")
    except pd.errors.EmptyDataError:
        raise SystemExit(f"Error: file is empty: {path}")

    total_rows, total_columns = frame.shape
    shown = frame.head(rows)

    info = f"File: {path}  ({total_rows} rows x {total_columns} columns)"
    _emit(info, limit)
    _emit(f"Showing first {len(shown)} row(s):", limit)
    print()
    print(render_table(shown, limit))

    if total_rows > rows:
        print()
        _emit(f"... {total_rows - rows} more row(s) not shown", limit)
    elif total_rows == 0:
        print()
        _emit("(file has no data rows)", limit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display the first rows of a CSV file in a table that "
        "wraps to the terminal width.",
    )
    parser.add_argument("csv_file", type=Path, help="path to the CSV file")
    parser.add_argument(
        "-n",
        "--rows",
        type=int,
        default=10,
        help="number of rows to display (default: 10)",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=None,
        help="total width in characters (default: terminal width)",
    )
    args = parser.parse_args()

    if args.rows < 1:
        raise SystemExit("Error: --rows must be 1 or greater")

    if args.width is None:
        limit = max(20, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
    else:
        limit = max(20, args.width)

    preview(args.csv_file, args.rows, limit)


if __name__ == "__main__":
    main()
