"""
Data downloader module for PhysioNet Challenge 2018 data.
"""

import re
import subprocess

import questionary
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    from .config import RAW_DIR
    from .utils import format_file_size
except ImportError:
    from config import RAW_DIR
    from utils import format_file_size


class PhysioNetDownloader:
    """Handles downloading data from PhysioNet."""

    def __init__(self, console):
        self.console = console
        self.base_url = "https://physionet.org/files/challenge-2018/1.0.0/training/"

    def check_wget_available(self) -> bool:
        """Check if wget is installed."""
        try:
            subprocess.run(
                ["which", "wget"],
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def download_menu(self) -> str:
        """Show download menu and return selection."""
        self.console.print("\n[bold cyan]PhysioNet Data Downloader[/bold cyan]")
        self.console.print(
            "[dim]Source: https://physionet.org/files/challenge-2018/1.0.0/[/dim]"
        )
        self.console.print(
            "[dim yellow]Note: PhysioNet may limit download speeds to ~1-2 MB/s[/dim yellow]\n"
        )

        if not self.check_wget_available():
            self.console.print(
                "[red]✗ wget is not installed. Please install it first:[/red]"
            )
            self.console.print("  macOS: brew install wget")
            self.console.print("  Linux: sudo apt-get install wget")
            return "back"

        download_type = questionary.select(
            "What would you like to download?",
            choices=[
                questionary.Choice("📦 Specific training records", value="specific"),
                questionary.Choice("📦 Range of training records", value="range"),
                questionary.Choice(
                    "📦 All training data (~1.5GB)", value="all_training"
                ),
                questionary.Choice("🔙 Back", value="back"),
            ],
        ).ask()

        return download_type

    def download_specific_records(self):
        """Download specific records by entering record IDs."""
        self.console.print("\n[bold]Download Specific Records[/bold]")
        self.console.print(
            "[dim]Enter record IDs (e.g., tr03-0146, tr03-0147) separated by commas[/dim]"
        )

        record_ids = questionary.text(
            "Record IDs:",
            default="tr03-0146",
        ).ask()

        if not record_ids:
            return

        # Parse record IDs
        records = [r.strip() for r in record_ids.split(",")]

        # Download each record
        for record in records:
            self.download_single_record(record)

        self.console.print(
            f"\n[green]✓[/green] Downloaded {len(records)} record(s) to {RAW_DIR}"
        )

    def download_range_records(self):
        """Download a range of records."""
        self.console.print("\n[bold]Download Range of Records[/bold]")
        self.console.print("[dim]Enter start and end record numbers[/dim]")

        start = questionary.text(
            "Start record (e.g., tr03-0100):", default="tr03-0100"
        ).ask()
        end = questionary.text(
            "End record (e.g., tr03-0110):", default="tr03-0110"
        ).ask()

        if not start or not end:
            return

        # Extract numbers from record IDs
        start_match = re.match(r"(tr\d+-?)(\d+)", start)
        end_match = re.match(r"(tr\d+-?)(\d+)", end)

        if not start_match or not end_match:
            self.console.print("[red]✗ Invalid record format[/red]")
            return

        prefix = start_match.group(1)
        start_num = int(start_match.group(2))
        end_num = int(end_match.group(2))

        if start_num > end_num:
            self.console.print("[red]✗ Start record must be less than end record[/red]")
            return

        # Download records in range
        records = []
        for num in range(start_num, end_num + 1):
            record = f"{prefix}{num:04d}"
            records.append(record)

        self.console.print(f"\n[bold]Downloading {len(records)} records...[/bold]")

        success_count = 0
        for record in records:
            if self.download_single_record(record, show_progress=False):
                success_count += 1

        self.console.print(
            f"\n[green]✓[/green] Successfully downloaded {success_count}/{len(records)} records to {RAW_DIR}"
        )

    def download_all_training(self):
        """Download all training data using wget recursive."""
        self.console.print("\n[yellow]⚠ This will download ~1.5GB of data[/yellow]")

        confirm = questionary.confirm("Continue with download?", default=False).ask()

        if not confirm:
            return

        self.console.print("\n[bold cyan]Downloading all training data...[/bold cyan]")

        # Create data directory
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        cmd = [
            "wget",
            "-r",  # recursive
            "-N",  # timestamping
            "-c",  # continue
            "-np",  # no parent
            "-nH",  # no host directories
            "--cut-dirs=4",  # cut directory depth
            "-P",
            str(RAW_DIR),  # output directory
            "-A",
            "*.mat,*.arousal,*.hea",  # accept only these file types
            self.base_url,
        ]

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                progress.add_task(description="Downloading...", total=None)
                result = subprocess.run(
                    cmd,
                    cwd=str(RAW_DIR.parent),
                    capture_output=False,
                )

            if result.returncode == 0:
                self.console.print(
                    f"\n[green]✓[/green] Download completed to {RAW_DIR}"
                )
            else:
                self.console.print(f"\n[red]✗[/red] Download failed")

        except KeyboardInterrupt:
            self.console.print("\n[yellow]✗ Download cancelled[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]✗ Error during download: {str(e)}[/red]")

    def download_single_record(self, record: str, show_progress: bool = True) -> bool:
        """Download a single record from PhysioNet.

        Args:
            record: Record ID (e.g., "tr03-0146")
            show_progress: Whether to show progress messages

        Returns:
            True if download successful, False otherwise
        """
        # Create record directory
        record_dir = RAW_DIR / record
        record_dir.mkdir(parents=True, exist_ok=True)

        # Files to download for each record
        extensions = [".mat", ".arousal", ".hea"]

        if show_progress:
            self.console.print(f"\n[bold]Downloading {record}...[/bold]")

        success = True
        for ext in extensions:
            url = f"{self.base_url}{record}/{record}{ext}"
            output_file = record_dir / f"{record}{ext}"

            # Skip if file already exists
            if output_file.exists():
                if show_progress:
                    self.console.print(f"  [dim]✓ {record}{ext} (already exists)[/dim]")
                continue

            try:
                # Use wget with progress bar
                cmd = ["wget", "--progress=bar:force", "-O", str(output_file), url]
                result = subprocess.run(
                    cmd, capture_output=False, timeout=900
                )  # 15 minutes timeout

                if result.returncode == 0 and output_file.exists():
                    if show_progress:
                        file_size = output_file.stat().st_size
                        size_str = format_file_size(file_size)
                        self.console.print(
                            f"  [green]✓[/green] {record}{ext} ({size_str})"
                        )
                else:
                    if show_progress:
                        self.console.print(f"  [red]✗[/red] {record}{ext} (failed)")
                    success = False
            except subprocess.TimeoutExpired:
                if show_progress:
                    self.console.print(
                        f"  [red]✗[/red] {record}{ext} (timeout after 15 minutes)"
                    )
                success = False
            except Exception as e:
                if show_progress:
                    self.console.print(f"  [red]✗[/red] {record}{ext} ({str(e)})")
                success = False

        return success
