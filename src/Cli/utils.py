"""
Utility functions for the Sleep Data Analysis Pipeline CLI.
"""

from pathlib import Path

from questionary import Style

try:
    from .config import PROCESSED_DIR, RAW_DIR
except ImportError:
    from config import PROCESSED_DIR, RAW_DIR


def format_file_size(size_bytes: float) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_record_path(record: str, data_source: str) -> Path:
    """Get the full path to a record directory."""
    if data_source == "raw":
        return RAW_DIR / record
    else:
        return PROCESSED_DIR / record


def set_custom_directory(console) -> str:
    """
    Prompt user for a custom data directory path.

    Returns:
        Path string or None if cancelled
    """
    import questionary

    console.print(
        "[bold cyan]📁 Enter custom directory path[/bold cyan]",
    )
    console.print(
        "[dim]Examples: /path/to/data/raw, C:\\Users\\Data, ./my_records[/dim]"
    )

    path_input = questionary.text(
        "Directory path:",
        validate=lambda text: bool(text.strip()),
    ).ask()

    if not path_input:
        console.print("[yellow]↩️ Using default directory[/yellow]")
        return None

    custom_path = Path(path_input.strip())

    if not custom_path.exists():
        console.print(f"[red]✗ Directory does not exist: {custom_path}[/red]")
        retry = questionary.confirm("🔄 Try another path?", default=True).ask()
        if retry:
            return set_custom_directory(console)
        else:
            console.print("[yellow]↩️ Using default directory[/yellow]")
            return None

    if not custom_path.is_dir():
        console.print(f"[red]✗ Path is not a directory: {custom_path}[/red]")
        retry = questionary.confirm("🔄 Try another path?", default=True).ask()
        if retry:
            return set_custom_directory(console)
        else:
            console.print("[yellow]↩️ Using default directory[/yellow]")
            return None

    console.print(f"[green]✓[/green] Using custom directory: {custom_path}")
    console.print(f"[dim]📁 Loading records from: {custom_path}[/dim]")
    return str(custom_path)


def get_questionary_style() -> Style:
    """Get questionary style configuration."""
    return Style(
        [
            ("qmark", "fg:#e91e63 bold"),
            ("question", "fg:#673ab7 bold"),
            ("answer", "fg:#2196f3 bold"),
            ("pointer", "fg:#e91e63 bold"),
            ("highlighted", "fg:#e91e63 bold"),
            ("selected", "fg:#4caf50"),
            ("separator", "fg:#999999"),
            ("instruction", "fg:#999999"),
        ]
    )
