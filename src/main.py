import questionary
from rich.console import Console

from Cli.utils import get_questionary_style

console = Console()

if __name__ == "__main__":
    running = True
    # Select program to run
    while running:
        console.print(
            "Welcome to the Sleep Data Analysis Pipeline!", style="bold green"
        )
        program_choice = questionary.select(
            "Select a program to run:",
            choices=[
                questionary.Choice("📊 CLI Interface", value="cli"),
                questionary.Choice(
                    "_TRAINING PIPELINE (NOT IMPLEMENTED YET)", value="training"
                ),
                questionary.Choice("❌ Exit", value="exit"),
            ],
            style=get_questionary_style(),
        ).ask()

        if program_choice == "cli":
            from Cli import SleepDataPipeline

            pipeline = SleepDataPipeline(console)
            pipeline.run()
        elif program_choice == "training":
            # TODO - implement training pipeline
            console.print(
                "Training pipeline is not implemented yet. Please select the CLI interface.",
                style="yellow",
            )
        elif program_choice == "exit":
            console.print("Goodbye!", style="bold green")
            running = False
