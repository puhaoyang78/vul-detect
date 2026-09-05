from __future__ import annotations

"""Command-line entrypoint for the staged vulnerability workflow."""

from .workflow import build_parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
