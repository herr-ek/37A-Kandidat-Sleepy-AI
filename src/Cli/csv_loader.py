"""
CSV loader for the Sleep Data Analysis Pipeline CLI.

Handles loading SpO2 / sleep data from CSV files (e.g. EmotiBit exports).
Normalises column names and timestamp units so the resulting DataFrame is
compatible with the rest of the pipeline (expects ``time_s`` in seconds
starting from 0 and ``sao2_percent``).
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import questionary
from rich import box
from rich.console import Console
from rich.table import Table

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR

# Default directory to scan for CSV files
CSV_DIR = DATA_DIR / "emotibit"

# Well-known column aliases → canonical name
_TIME_ALIASES = {"timestamp", "time", "time_ms", "time_s", "t", "time_sec"}
_SPO2_ALIASES = {"o2", "spo2", "sao2", "sao2_percent", "spo2_percent", "o2_percent"}


class CsvLoader:
    """Interactive CSV file loader for SpO2 / sleep data."""

    def __init__(self, console: Console, style):
        self.console = console
        self.style = style

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Optional[tuple[pd.DataFrame, str]]:
        """
        Interactively select and load a CSV file.

        Returns:
            ``(df, record_name)`` where *df* has at minimum ``time_s`` and
            ``sao2_percent`` columns, or ``None`` if the user cancels.
        """
        path = self._pick_file()
        if path is None:
            return None

        try:
            raw = pd.read_csv(path)
        except Exception as exc:
            self.console.print(f"[red]✗ Could not read CSV: {exc}[/red]")
            return None

        self._preview(raw, path.name)

        time_col = self._pick_column(raw, "time / timestamp", _TIME_ALIASES)
        if time_col is None:
            return None

        spo2_col = self._pick_column(raw, "SpO2 / O2 percentage", _SPO2_ALIASES)
        if spo2_col is None:
            return None

        time_unit = self._pick_time_unit(raw[time_col])

        df = self._build_dataframe(raw, time_col, spo2_col, time_unit)
        record_name = path.stem

        self.console.print(
            f"[green]✓[/green] Loaded [bold]{path.name}[/bold]  "
            f"→ {len(df):,} rows  |  "
            f"{df['time_s'].iloc[0]:.1f} – {df['time_s'].iloc[-1]:.1f} s"
        )
        return df, record_name

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    def _pick_file(self) -> Optional[Path]:
        csv_files = sorted(CSV_DIR.glob("*.csv")) if CSV_DIR.exists() else []

        choices = [questionary.Choice(p.name, value=p) for p in csv_files]
        choices.append(questionary.Choice("📂 Enter custom file path…", value="custom"))
        if choices:
            choices.append(questionary.Choice("❌ Cancel", value=None))

        if not choices or (len(choices) == 2 and choices[0].value == "custom"):
            # No CSVs found in default dir — go straight to custom path
            return self._prompt_custom_path()

        selected = questionary.select(
            f"Select a CSV file  (found {len(csv_files)} in data/emotibit/):",
            choices=choices,
            style=self.style,
        ).ask()

        if selected is None:
            return None
        if selected == "custom":
            return self._prompt_custom_path()
        return selected

    def _prompt_custom_path(self) -> Optional[Path]:
        raw = questionary.path(
            "Enter path to CSV file:",
            style=self.style,
        ).ask()
        if not raw:
            return None
        p = Path(raw).expanduser()
        if not p.exists():
            self.console.print(f"[red]✗ File not found: {p}[/red]")
            return None
        return p

    # ------------------------------------------------------------------
    # Column selection
    # ------------------------------------------------------------------

    def _preview(self, df: pd.DataFrame, filename: str) -> None:
        table = Table(title=f"CSV preview — {filename}", box=box.SIMPLE)
        for col in df.columns:
            table.add_column(col, style="cyan")
        for _, row in df.head(5).iterrows():
            table.add_row(*[str(v) for v in row])
        self.console.print(table)

    def _pick_column(
        self,
        df: pd.DataFrame,
        description: str,
        aliases: set[str],
    ) -> Optional[str]:
        """Return a column name, auto-detecting if possible, else asking the user."""
        # Try case-insensitive alias match
        for col in df.columns:
            if col.strip().lower() in aliases:
                self.console.print(
                    f"[dim]  Auto-detected [bold]{col}[/bold] as {description}[/dim]"
                )
                return col

        # Ask user
        cols = df.columns.tolist()
        selected = questionary.select(
            f"Which column is the {description}?",
            choices=cols,
            style=self.style,
        ).ask()
        return selected

    # ------------------------------------------------------------------
    # Timestamp handling
    # ------------------------------------------------------------------

    def _pick_time_unit(self, time_series: pd.Series) -> str:
        """Heuristically detect the timestamp unit and let the user confirm."""
        median_val = time_series.median()

        # Heuristic thresholds:
        #   < 1e4  →  likely already seconds
        #   < 1e7  →  likely milliseconds
        #   else   →  likely microseconds
        if median_val < 1e4:
            guess = "seconds"
        elif median_val < 1e7:
            guess = "milliseconds"
        else:
            guess = "microseconds"

        unit = questionary.select(
            f"Timestamp unit?  (median value ≈ {median_val:.0f} → guessing {guess})",
            choices=[
                questionary.Choice("Seconds  (no conversion needed)", value="s"),
                questionary.Choice("Milliseconds  (÷ 1 000)", value="ms"),
                questionary.Choice("Microseconds  (÷ 1 000 000)", value="us"),
            ],
            default={"seconds": "s", "milliseconds": "ms", "microseconds": "us"}[guess],
            style=self.style,
        ).ask()
        return unit or "ms"

    # ------------------------------------------------------------------
    # DataFrame construction
    # ------------------------------------------------------------------

    def _build_dataframe(
        self,
        raw: pd.DataFrame,
        time_col: str,
        spo2_col: str,
        time_unit: str,
    ) -> pd.DataFrame:
        divisor = {"s": 1, "ms": 1_000, "us": 1_000_000}[time_unit]

        time_s = raw[time_col].astype(float) / divisor
        # Make relative to recording start
        time_s = time_s - time_s.iloc[0]

        df = pd.DataFrame(
            {
                "time_s": time_s.values,
                "sao2_percent": raw[spo2_col].astype(float).values,
            }
        )
        df = df.sort_values("time_s").reset_index(drop=True)
        return df
