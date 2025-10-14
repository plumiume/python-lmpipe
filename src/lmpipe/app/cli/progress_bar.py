from typing import TypedDict
from rich import progress

# batch progress

class BatchProgressColumns(TypedDict):
    status: progress.TextColumn
    spinner: progress.SpinnerColumn
    bar: progress.BarColumn
    percentage: progress.MofNCompleteColumn
    elapsed: progress.TimeElapsedColumn
    _time_sep: progress.TextColumn
    remaining: progress.TimeRemainingColumn

batch_progress_columns: BatchProgressColumns = {
    'status': progress.TextColumn("[progress.description]{task.description}"),
    'spinner': progress.SpinnerColumn(),
    'bar': progress.BarColumn(bar_width=None),
    'percentage': progress.MofNCompleteColumn(),
    'elapsed': progress.TimeElapsedColumn(),
    '_time_sep': progress.TextColumn("/"),
    'remaining': progress.TimeRemainingColumn(),
}

batch_progress = progress.Progress(
    *(
        col for col in batch_progress_columns.values()
        if isinstance(col, progress.ProgressColumn)
    )
)

# sample progress

class SampleProgressColumns(TypedDict):
    description: progress.TextColumn
    bar: progress.BarColumn
    percentage: progress.MofNCompleteColumn
    elapsed: progress.TimeElapsedColumn
    _time_sep: progress.TextColumn
    remaining: progress.TimeRemainingColumn

sample_progress_columns: SampleProgressColumns = {
    'description': progress.TextColumn("[progress.description]{task.description}"),
    'bar': progress.BarColumn(bar_width=None),
    'percentage': progress.MofNCompleteColumn(),
    'elapsed': progress.TimeElapsedColumn(),
    '_time_sep': progress.TextColumn("/"),
    'remaining': progress.TimeRemainingColumn(),
}

sample_progress = progress.Progress(
    *(
        col for col in sample_progress_columns.values()
        if isinstance(col, progress.ProgressColumn)
    )
)
