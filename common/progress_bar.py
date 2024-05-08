import sys


def put(
    fraction,
    num_progress_chars=10,
    prefix="Progress: ",
    show_percentage=True,
    end='',
    file=sys.stdout,
    flush=True):
    """
    Output a progress bar.
    """
    assert 0.0 <= fraction and fraction <= 1.0
    
    result = f"\r{prefix}["

    # Draw the bar
    num_completed_chars = int(num_progress_chars * fraction + 0.5)
    result += "█" * num_completed_chars
    result += "_" * (num_progress_chars - num_completed_chars)

    result += "]"

    if show_percentage:
        result += f" {fraction * 100:6.2f}%"

    print(result, end=end, file=file, flush=flush)
