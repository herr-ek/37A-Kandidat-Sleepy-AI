"""
Results Browser for the Sleep Data Analysis Pipeline CLI.

Lets the user pick CSV result files from batch jobs, browse models in a
sortable table, and drill into a detail view with parameters, metrics, and a
confusion matrix visualisation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import questionary
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR

RESULTS_DIR = DATA_DIR.parent / "jobs" / "results"

METRIC_COLUMNS = ["Balanced Accuracy", "Accuracy", "Recall", "F1 Macro"]
CM_COLUMNS = ["TN", "FP", "FN", "TP"]


def _split_hyperparams(hp_str: str) -> list[str]:
    """Split a hyperparameter string on ', ' but ignore commas inside brackets."""
    items, current, depth = [], [], 0
    i = 0
    while i < len(hp_str):
        ch = hp_str[i]
        if ch in "([":
            depth += 1
            current.append(ch)
        elif ch in ")]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0 and hp_str[i : i + 2] == ", ":
            items.append("".join(current).strip())
            current = []
            i += 1  # skip the space too
        else:
            current.append(ch)
        i += 1
    if current:
        items.append("".join(current).strip())
    return items


class ResultsBrowser:
    """Interactive browser for batch-job CSV result files."""

    def __init__(self, console: Console, style):
        self.console = console
        self.style = style

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    def run(self):
        self.console.print(
            "\n[bold cyan]─── Results Browser ──────────────────────────────[/bold cyan]"
        )

        df = self._load_csv_files()
        if df is None or df.empty:
            return

        # Outer loop: re-enter when user presses ESC in model picker (= change sort)
        sort_by = METRIC_COLUMNS[0]
        while True:
            sort_by = self._pick_sort(df, sort_by)
            if sort_by is None:
                return
            # Inner loop: pick model → detail view → back to picker
            while True:
                self._render_summary_table(df, sort_by)
                chosen_row = self._pick_model(df, sort_by)
                if chosen_row is None:  # ESC pressed → go back to sort
                    break
                if isinstance(chosen_row, str) and chosen_row == "exit":
                    return
                self._show_detail(chosen_row)

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def _load_csv_files(self) -> pd.DataFrame | None:
        csv_files = sorted(RESULTS_DIR.glob("*.csv")) if RESULTS_DIR.exists() else []

        if not csv_files:
            # Let user point to a directory
            custom = questionary.path(
                "No CSVs found in jobs/results/. Enter a directory to search:",
                style=self.style,
            ).ask()
            if not custom:
                return None
            csv_files = sorted(Path(custom).glob("*.csv"))

        if not csv_files:
            self.console.print("[red]✗ No CSV files found.[/red]")
            return None

        choices = [questionary.Choice(f.name, value=f) for f in csv_files] + [
            questionary.Choice("✅ Done selecting", value=None)
        ]

        selected = questionary.checkbox(
            "Select result CSV files to load:",
            choices=[questionary.Choice(f.name, value=f) for f in csv_files],
            style=self.style,
        ).ask()

        if not selected:
            self.console.print("[yellow]✗ No files selected.[/yellow]")
            return None

        frames = []
        for path in selected:
            try:
                frames.append(pd.read_csv(path))
                self.console.print(
                    f"[dim]Loaded {path.name} ({len(frames[-1])} rows)[/dim]"
                )
            except Exception as exc:
                self.console.print(
                    f"[yellow]⚠ Could not read {path.name}: {exc}[/yellow]"
                )

        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        # Coerce metric columns to float (they may be strings from pre-formatted CSVs)
        for col in METRIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        self.console.print(
            f"[green]✓[/green] Loaded [bold]{len(df)}[/bold] model results."
        )
        return df

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def _pick_sort(self, df: pd.DataFrame, current_sort: str) -> str:
        """Prompt the user to choose a sort column."""
        available_metrics = [c for c in METRIC_COLUMNS if c in df.columns]
        sort_by = questionary.select(
            "Sort by:",
            choices=[
                questionary.Choice(f"{m}{' ✓' if m == current_sort else ''}", value=m)
                for m in available_metrics
            ],
            default=current_sort,
            style=self.style,
        ).ask()
        return sort_by

    def _render_summary_table(self, df: pd.DataFrame, sort_by: str):
        sorted_df = df.sort_values(by=sort_by, ascending=False, na_position="last")

        table = Table(title=f"Model Results  (sorted by {sort_by})", box=box.ROUNDED)
        table.add_column("#", style="dim", justify="right")
        table.add_column("Model", style="green", width=40)
        table.add_column("Type", style="cyan")

        metric_cols = [c for c in METRIC_COLUMNS if c in df.columns]
        for col in metric_cols:
            style = "bold yellow" if col == sort_by else "white"
            table.add_column(col, justify="right", style=style)

        for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
            cells = [
                str(rank),
                str(row.get("Model File", "?")),
                str(row.get("Model Type", "?")),
            ]
            for col in metric_cols:
                val = row.get(col)
                cells.append(f"{val:.4f}" if pd.notna(val) else "N/A")
            table.add_row(*cells)

        self.console.print(table)

    # ------------------------------------------------------------------
    # Model picker
    # ------------------------------------------------------------------

    def _pick_model(self, df: pd.DataFrame, sort_by: str):
        """Autocomplete model selection.

        Returns:
            pd.Series  — the selected model row
            None       — ESC was pressed → caller should change sort order
            "exit"     — user chose to leave the browser
        """
        _RESORT = "← Change sort order"
        _EXIT = "← Exit, Back to main menu"

        sorted_df = df.sort_values(by=sort_by, ascending=False, na_position="last")
        # Preserve sort order in the completion list
        name_to_idx = {
            str(row.get("Model File", f"model_{idx}")): idx
            for idx, row in sorted_df.iterrows()
        }
        all_choices = list(name_to_idx.keys()) + [_RESORT, _EXIT]

        chosen = questionary.autocomplete(
            "Type to filter and select a model  [^C = change sort]:",
            choices=all_choices,
            style=self.style,
            validate=lambda x: x in all_choices
            or "Please select an entry from the list",
        ).ask()

        if chosen is None or chosen == _RESORT:
            return None
        if isinstance(chosen, str) and chosen == _EXIT:
            return "exit"

        idx = name_to_idx.get(chosen)
        if idx is None:
            return None
        return df.loc[idx]

    # ------------------------------------------------------------------
    # Detail view
    # ------------------------------------------------------------------

    def _show_detail(self, row: pd.Series):
        model_name = row.get("Model File", "Unknown")

        self.console.print(
            Panel(
                f"[bold]{model_name}[/bold]  —  {row.get('Model Type', '?')}",
                title="Model Detail",
                border_style="cyan",
            )
        )

        # — Hyperparameters ——————————————————————————————————————
        hp_str = row.get("Hyperparameters", "")
        if hp_str and pd.notna(hp_str):
            hp_table = Table(title="Hyperparameters", box=box.SIMPLE)
            hp_table.add_column("Parameter", style="cyan")
            hp_table.add_column("Value", style="white")
            for item in _split_hyperparams(str(hp_str)):
                if "=" in item:
                    k, _, v = item.partition("=")
                    hp_table.add_row(k.strip(), v.strip())
            self.console.print(hp_table)

        # — Metrics ——————————————————————————————————————————————
        metric_cols = [c for c in METRIC_COLUMNS if c in row.index]
        if metric_cols:
            m_table = Table(title="Evaluation Metrics", box=box.ROUNDED)
            m_table.add_column("Metric", style="cyan")
            m_table.add_column("Value", justify="right", style="green")
            for col in metric_cols:
                val = row.get(col)
                m_table.add_row(col, f"{val:.4f}" if pd.notna(val) else "N/A")
            self.console.print(m_table)

        # — Confusion matrix ————————————————————————————————————
        cm_present = all(c in row.index and pd.notna(row.get(c)) for c in CM_COLUMNS)
        if cm_present:
            tn = float(row["TN"])
            fp = float(row["FP"])
            fn = float(row["FN"])
            tp = float(row["TP"])

            cm_table = Table(title="Confusion Matrix (normalised)", box=box.SIMPLE)
            cm_table.add_column("", style="dim")
            cm_table.add_column("Predicted 1", style="red", justify="right")
            cm_table.add_column("Predicted 0", style="green", justify="right")
            cm_table.add_row("Actual 1", f"{tp:.4f}", f"{fn:.4f}")
            cm_table.add_row("Actual 0", f"{fp:.4f}", f"{tn:.4f}")
            self.console.print(cm_table)

            # ASCII bar chart to make the CM more visual
            self.console.print(self._cm_bars(tn, fp, fn, tp))

        # — Metadata ————————————————————————————————————————————
        saved_at = row.get("Saved At", "")
        num_rec = row.get("Num Records", "")
        if pd.notna(saved_at) or pd.notna(num_rec):
            meta_table = Table(box=box.SIMPLE)
            meta_table.add_column("Field", style="dim")
            meta_table.add_column("Value", style="white")
            if pd.notna(saved_at):
                meta_table.add_row("Saved at", str(saved_at))
            if pd.notna(num_rec):
                meta_table.add_row("Training records", str(int(num_rec)))
            self.console.print(meta_table)

        choices = [
            questionary.Choice("Back to model list", value="back"),
            questionary.Choice("Print confusion matrix", value="print_cm"),
        ]  # Add more actions here
        while True:
            action = questionary.select(
                "Choose an action:",
                choices=choices,
                style=self.style,
            ).ask()
            if action == "back":
                break
            elif action == "print_cm" and cm_present:
                cm_array = np.array([[tp, fn], [fp, tn]])

                # Per-cell colour: diagonal = green (good), off-diagonal = red (bad)
                import matplotlib.colors as mcolors

                good = np.array(mcolors.to_rgba("#40eb87"))  # emerald green
                bad = np.array(mcolors.to_rgba("#dd5c4e"))  # alizarin red
                white = np.array([1.0, 1.0, 1.0, 1.0])
                rgba = np.zeros((2, 2, 4))
                for i in range(2):
                    for j in range(2):
                        base = good if i == j else bad
                        rgba[i, j] = white + cm_array[i, j] * (base - white)

                fig, ax = plt.subplots(figsize=(5, 4))
                ax.imshow(rgba, aspect="equal")
                for (i, j), val in np.ndenumerate(cm_array):
                    brightness = (
                        0.299 * rgba[i, j, 0]
                        + 0.587 * rgba[i, j, 1]
                        + 0.114 * rgba[i, j, 2]
                    )
                    text_color = "black" if brightness > 0.55 else "white"
                    ax.text(
                        j,
                        i,
                        f"{val:.4f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=13,
                        fontweight="bold",
                    )
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Predicted 1", "Predicted 0"])
                ax.set_yticklabels(["Actual 1", "Actual 0"])
                plt.title(f"Confusion Matrix — {model_name}", pad=12)
                plt.tight_layout()
                plt.show()
            else:
                break
        # questionary.press_any_key_to_continue("Press any key to go back…").ask()

    @staticmethod
    def _cm_bars(tn: float, fp: float, fn: float, tp: float) -> str:
        """Return a compact ASCII bar visualisation of the confusion matrix cells."""
        width = 30
        rows = [
            ("TN (correct normal)", tn),
            ("FP (false alarm)", fp),
            ("FN (missed apnea)", fn),
            ("TP (correct apnea)", tp),
        ]
        lines = ["", "[dim]Distribution:[/dim]"]
        for label, val in rows:
            filled = round(val * width)
            bar = "█" * filled + "░" * (width - filled)
            pct = f"{100 * val:.1f}%"
            lines.append(f"  [dim]{label:<22}[/dim] [cyan]{bar}[/cyan] {pct}")
        lines.append("")
        return "\n".join(lines)
