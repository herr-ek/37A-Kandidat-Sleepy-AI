# Architecture Overview

## Module Dependency Graph

```
                         ┌─────────────────────────────────┐
                         │     cli.py (Orchestrator)       │
                         │    SleepDataPipeline class      │
                         └──────────────┬──────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
            ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
            │   config.py  │    │   utils.py   │    │    ui.py     │
            │  Constants & │    │  Utilities   │    │  User Input  │
            │    Paths     │    │  & Helpers   │    │   & Prompts  │
            └──────────────┘    └──────────────┘    └──────────────┘
                                       ▲                       ▲
                    ┌──────────────────┴───────────────────────┘
                    │
        ┌───────────┴────────────┬──────────────┬──────────────┬────────────────┐
        │                        │              │              │                │
        ▼                        ▼              ▼              ▼                ▼
    ┌──────────────┐    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐
    │ data_loader  │    │ data_display │  │data_processor│  │ data_saver   │  │  downloader.py  │
    │   (Loading)  │    │ (Visualization)  │(Processing) │  │  (Saving)    │  │ (PhysioNet DL)  │
    └──────────────┘    └──────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘
         │                                       │                  │
         │                                       │                  │
    ┌────┴────┐              ┌─────────────────┴────┐          ┌───┴───┐
    │ External │              │    External          │          │External│
    │Libraries │              │     Libraries        │          │Library │
    │          │              │                      │          │        │
    │ pandas   │              │ pandas, numpy,       │          │ pandas │
    │ pathlib  │              │ preprocessing,       │          │        │
    │ DMlib    │              │ resampling, features │          │ shutil │
    └──────────┘              └──────────────────────┘          └────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │  batch_processor.py      │
                         │  (Batch Operations)      │
                         │  (Uses loader, processor)│
                         └──────────────────────────┘
```

## Control Flow Diagram

```
                    START (cli.py main())
                              │
                              ▼
                      run() - Main loop
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
              Choose       Select      Choose
              Data Source  Records     Action
                (UI)         (UI)        (UI)
                │             │           │
                ▼             ▼           ▼
            DataLoader   DataLoader   execute_action()
                │             │           │
                │             │      ┌────┼─────────────────────┐
                │             │      │    │    │    │    │      │
                │             │      ▼    ▼    ▼    ▼    ▼      ▼
                │             │     Display Stats Process Save Export
                │             │     (DataDisplay)  (DataProcessor) (DataSaver)
                │             │
                │             └────────────────────⬇
                │                                   │
                └───────────────────────────────────┼───────┐
                                                    │       │
                                                    ▼       ▼
                                             Load complete  Loop?
                                                    │       │
                                                    │    ┌──┴──┐
                                                    │    │ Yes │ -> back to Choose Action
                                                    │    └─────┘
                                                    │ No
                                                    ▼
                                                   END
```

## Module Interactions

### Single Mode Workflow
```
User                 CLI              Components
 │                   │                    │
 │─ Select Raw/Proc─→│                    │
 │                   │─ UI Menu ────────→ ui.py
 │                   │                    │
 │─ Select Record ──→│─ Load Records ──→ data_loader.py
 │                   │                    │
 │                   │─ Display ────────→ data_display.py
 │                   │                    │
 │─ Choose Action ──→│─ UI Menu ────────→ ui.py
 │                   │                    │
 │                   │─ Process ────────→ data_processor.py
 │                   │      or            or
 │                   │      Save ────────→ data_saver.py
 │                   │                    │
 └────────────────────────┬───────────────┘
                          │
                    (Loop or Exit)
```

### Batch Mode Workflow
```
User                 CLI              Components
 │                   │                    │
 │─ Batch Mode ─────→│─ UI Menu ────────→ ui.py
 │                   │                    │
 │─ Select Records ──→│─ Load Records ──→ data_loader.py
 │                   │                    │
 │─ Batch Action ───→│─ Batch Proc ────→ batch_processor.py
 │                   │       │            │
 │                   │       └─ Loop ────→ For each record:
 │                   │                      - Load (data_loader)
 │                   │                      - Process (data_processor)
 │                   │                      - Save (auto)
 │                   │                    │
 └────────────────────────┬───────────────┘
                          │
                    (Menu or Exit)
```

## Data Flow: Important Objects

```
SleepDataPipeline Instance
├── data_source: "raw" | "processed"
├── mode: "single" | "batch"
├── selected_records: List[str]
├── custom_data_dir: Optional[str]
└── Component Modules:
    ├── DataLoader
    │   ├── current_dataframe: pd.DataFrame
    │   └── applied_operations: List[str]
    ├── DataDisplay (stateless)
    ├── DataProcessor (stateless)
    ├── DataSaver (stateless)
    ├── PhysioNetDownloader (stateless)
    └── BatchProcessor (stateless)
```

## Key Design Principles Applied

1. **Separation of Concerns**: Each module has one responsibility
2. **Single Responsibility**: Classes do one thing well
3. **Dependency Injection**: Console and config passed to modules
4. **Statelessness**: Most components don't maintain state (only DataLoader and CLI do)
5. **Composition**: CLI composes components rather than inheriting
6. **Clear Interfaces**: Public methods are simple and well-defined
