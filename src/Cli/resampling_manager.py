"""
Resampling Manager for the Sleep Data Analysis Pipeline CLI.

Handles interactive resampling of already-processed records stored in
data/processed/.  Supports two strategies:

  * Interpolation  — existing ``resample_to_time_resolution``
  * Averaging      — new ``resample_average`` (mean of N samples per bin)
"""

import pyarrow.parquet as pq
import questionary
from rich import box
from rich.console import Console
from rich.table import Table

try:
    from .config import PROCESSED_DIR
except ImportError:
    from config import PROCESSED_DIR

try:
    from ..Resampling.Resampling import resample_average, resample_to_time_resolution
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Resampling.Resampling import resample_average, resample_to_time_resolution


class ResamplingManager:
    """Interactive resampling of processed parquet records."""

    def __init__(self, console: Console, style):
        self.console = console
        self.style = style

    # ------------------------------------------------------------------
    # Top-level entry point called from cli.py
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.console.print(
            "\n[bold cyan]─── Resampling ───────────────────────────────────[/bold cyan]"
        )

        records = self._find_processed_records()
        if not records:
            self.console.print(
                f"[red]✗ No processed parquet files found in {PROCESSED_DIR}[/red]"
            )
            return

        selected = self._pick_records(records)
        if not selected:
            return

        resolution = self._pick_resolution()
        if resolution is None:
            return

        method = self._pick_method()
        if method is None:
            return

        suffix = self._pick_output_suffix(resolution)
        if suffix is None:
            return

        self._run_resampling(selected, resolution, method, suffix)

    # ------------------------------------------------------------------
    # Record discovery
    # ------------------------------------------------------------------

    def _find_processed_records(self) -> list[dict]:
        records = []
        for record_dir in sorted(PROCESSED_DIR.iterdir()):
            if not record_dir.is_dir():
                continue
            processed_file = record_dir / f"{record_dir.name}_processed.parquet"
            if not processed_file.exists():
                continue
            try:
                meta = pq.read_metadata(processed_file)
                records.append(
                    {
                        "name": record_dir.name,
                        "path": processed_file,
                        "n_rows": meta.num_rows,
                    }
                )
            except Exception:
                pass
        return records

    # ------------------------------------------------------------------
    # Interactive prompts
    # ------------------------------------------------------------------

    def _pick_records(self, records: list[dict]) -> list[dict]:
        choices = [
            questionary.Choice(f"{r['name']}  ({r['n_rows']:,} rows)", value=r)
            for r in records
        ]
        selected = questionary.checkbox(
            "Select records to resample (space to toggle, enter to confirm):",
            choices=choices,
            style=self.style,
        ).ask()
        return selected or []

    def _pick_resolution(self) -> float | None:
        presets = ["0.5", "1.0", "2.0", "5.0", "Other…"]
        choice = questionary.select(
            "Target time resolution (seconds per sample):",
            choices=presets,
            style=self.style,
        ).ask()
        if choice is None:
            return None
        if choice == "Other…":
            raw = questionary.text(
                "Enter resolution in seconds (e.g. 0.25):", style=self.style
            ).ask()
            if raw is None:
                return None
            try:
                return float(raw)
            except ValueError:
                self.console.print("[red]✗ Invalid number.[/red]")
                return None
        return float(choice)

    def _pick_method(self) -> str | None:
        return questionary.select(
            "Resampling method:",
            choices=[
                questionary.Choice(
                    "Average  — mean of all samples in each bin (no data fabricated)",
                    value="average",
                ),
                questionary.Choice(
                    "Interpolate  — linear interpolation onto a new time grid",
                    value="interpolate",
                ),
            ],
            style=self.style,
        ).ask()

    def _pick_output_suffix(self, resolution: float) -> str | None:
        default = f"_resampled_{resolution}s"
        raw = questionary.text(
            "Output file suffix (appended to record name):",
            default=default,
            style=self.style,
        ).ask()
        return raw if raw is not None else None

    # ------------------------------------------------------------------
    # Core resampling loop
    # ------------------------------------------------------------------

    def _run_resampling(
        self,
        records: list[dict],
        resolution: float,
        method: str,
        suffix: str,
    ) -> None:
        import pandas as pd

        table = Table(title="Resampling Results", box=box.ROUNDED)
        table.add_column("Record", style="cyan")
        table.add_column("Original rows", justify="right")
        table.add_column("Resampled rows", justify="right", style="green")
        table.add_column("Output file", style="dim")

        for r in records:
            name = r["name"]
            try:
                df = pd.read_parquet(r["path"])
                orig_rows = len(df)

                if method == "average":
                    resampled = resample_average(df, resolution)
                else:
                    resampled = resample_to_time_resolution(df, resolution)

                out_path = PROCESSED_DIR / name / f"{name}{suffix}.parquet"
                resampled.to_parquet(out_path, index=False)

                table.add_row(
                    name,
                    f"{orig_rows:,}",
                    f"{len(resampled):,}",
                    out_path.name,
                )
            except Exception as exc:
                table.add_row(name, "—", "—", f"[red]Error: {exc}[/red]")

        self.console.print(table)
        self.console.print(
            f"[green]✓[/green] Resampled {len(records)} record(s) "
            f"to {resolution} s/sample using '{method}'."
        )
