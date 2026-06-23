from __future__ import annotations


MESSAGE = """
scripts/build_ch05_section51_notebook.py is retired.

The Section 5.1 tutorial notebook is maintained directly at:
  notebooks/chapter5_1_timecourse_suite.ipynb

The retired notebooks/05_1_single_cell_timecourse_main_suite.ipynb active copy
has been removed from notebooks/.

This entry point intentionally does not generate or overwrite the notebook.
""".strip()


def main() -> None:
    raise SystemExit(MESSAGE)


if __name__ == "__main__":
    main()
