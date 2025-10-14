"""Standard error capture utility for LMPipe.

This module provides a context manager for capturing standard error output
during pipeline execution, which is useful for logging and debugging purposes.
"""

from typing import Iterator, Any, TextIO, Iterable
from abc import ABC
import sys
import io
import os
import threading

class CaptureStderr(TextIO, ABC):
    """Context manager for capturing standard error output.
    
    This class captures stderr output by temporarily redirecting it to a StringIO
    buffer, allowing the captured content to be read while the context is active.
    Note that this implementation operates in read-only mode, so direct writing
    methods will raise OSError.
    
    Example:
        >>> with CaptureStderr() as capture:
        ...     # Reading captured content works
        ...     captured_text = capture.getvalue()
        ...     # Direct writing would raise OSError
        ...     # capture.write("text")  # raises OSError("not writable")
    """

    def __init__(self):
        """Initialize the stderr capture mechanism and start capturing."""
        self.original_stderr: Any = sys.stderr
        self.captured_output = io.StringIO()
        self._lock = threading.Lock()
        self._read_fd: int | None = None
        self._write_fd: int | None = None
        self._reader_thread: threading.Thread | None = None
        self._stop_reading = threading.Event()

        # Create a pipe for capturing stderr
        self._read_fd, self._write_fd = os.pipe()

        # Create TextIOWrapper for the write end of the pipe
        write_buffer = io.BufferedWriter(io.FileIO(self._write_fd, 'wb', closefd=False))
        sys.stderr = io.TextIOWrapper(
            write_buffer,
            encoding=sys.stderr.encoding,
            errors=sys.stderr.errors,
            newline=sys.stderr.newlines,
            line_buffering=bool(sys.stderr.line_buffering),
            write_through=getattr(sys.stderr, 'write_through', False)
        )

        # Start a thread to read from the pipe and capture the output
        self._reader_thread = threading.Thread(target=self._read_pipe, daemon=True)
        self._reader_thread.start()

    def __enter__(self) -> 'CaptureStderr':
        """Enter the context and return self."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Exit the context and stop capturing stderr.
        
        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        with self._lock:
            # Stop the reader thread
            self._stop_reading.set()
            
            # Close and restore original stderr
            if sys.stderr != self.original_stderr:
                try:
                    sys.stderr.close()
                except Exception:
                    pass
            
            if self.original_stderr is not None:
                sys.stderr = self.original_stderr
            
            # Close the pipe file descriptors
            if self._write_fd is not None:
                try:
                    os.close(self._write_fd)
                except Exception:
                    pass
                self._write_fd = None
            
            if self._read_fd is not None:
                try:
                    os.close(self._read_fd)
                except Exception:
                    pass
                self._read_fd = None
            
            # Wait for the reader thread to finish
            if self._reader_thread and self._reader_thread.is_alive():
                self._reader_thread.join(timeout=1.0)

    def _read_pipe(self) -> None:
        """Read data from the pipe in a separate thread."""
        if self._read_fd is None:
            return
            
        try:
            while not self._stop_reading.is_set():
                try:
                    # Use os.read with a timeout-like approach
                    # Read raw bytes from the pipe
                    data = os.read(self._read_fd, 1024)
                    if data:
                        # Decode bytes to string
                        text = data.decode('utf-8', errors='replace')
                        # Write to captured output
                        self.captured_output.write(text)
                        # Also write to original stderr
                        if self.original_stderr is not None:
                            try:
                                self.original_stderr.write(text)
                                self.original_stderr.flush()
                            except Exception:
                                pass
                    else:
                        # EOF reached
                        break
                except OSError:
                    # Pipe was closed or no data available
                    # Sleep briefly to avoid busy waiting
                    import time
                    time.sleep(0.01)
                    continue
                except Exception:
                    # Handle any other errors
                    break
        except Exception:
            # Handle any setup errors
            pass



    # TextIO interface implementation
    def close(self) -> None:
        """Close the capture mechanism."""
        pass

    @property
    def closed(self) -> bool:
        """Check if the capture is closed."""
        return False

    def fileno(self) -> int:
        """Return file descriptor (not applicable for StringIO)."""
        raise OSError("fileno() not supported")

    def flush(self) -> None:
        """Flush the captured output (no-op in read mode)."""
        # In read mode, flush is typically a no-op
        pass

    def isatty(self) -> bool:
        """Check if connected to a TTY."""
        return False

    def read(self, size: int = -1) -> str:
        """Read from captured output.
        
        Args:
            size (int): Number of characters to read.
            
        Returns:
            str: The read content.
        """
        return self.captured_output.read(size)

    def readable(self) -> bool:
        """Check if readable."""
        return True

    def readline(self, size: int = -1) -> str:
        """Read a line from captured output.
        
        Args:
            size (int): Maximum number of characters to read.
            
        Returns:
            str: The read line.
        """
        return self.captured_output.readline(size)

    def readlines(self, hint: int = -1) -> list[str]:
        """Read lines from captured output.
        
        Args:
            hint (int): Hint for number of lines.
            
        Returns:
            list[str]: List of lines.
        """
        return self.captured_output.readlines(hint)

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to position in captured output.
        
        Args:
            offset (int): Offset position.
            whence (int): Seek mode.
            
        Returns:
            int: New position.
        """
        return self.captured_output.seek(offset, whence)

    def seekable(self) -> bool:
        """Check if seekable."""
        return True

    def tell(self) -> int:
        """Get current position in captured output.
        
        Returns:
            int: Current position.
        """
        return self.captured_output.tell()

    def truncate(self, size: int | None = None) -> int:
        """Truncate captured output (not supported in read mode).
        
        Args:
            size (int | None): Size to truncate to.
            
        Returns:
            int: New size.
            
        Raises:
            OSError: Always raised as truncating is not supported in read mode.
        """
        raise OSError("not writable")

    def writable(self) -> bool:
        """Check if writable."""
        return False

    def write(self, s: str) -> int:
        """Write to captured output (not supported in read mode).
        
        Args:
            s (str): String to write.
            
        Returns:
            int: Number of characters written.
            
        Raises:
            OSError: Always raised as writing is not supported in read mode.
        """
        raise OSError("not writable")

    def writelines(self, lines: Iterable[str]) -> None:
        """Write lines to captured output (not supported in read mode).
        
        Args:
            lines (Iterable[str]): Lines to write (iterable of strings).
            
        Raises:
            OSError: Always raised as writing is not supported in read mode.
        """
        raise OSError("not writable")

    def __iter__(self) -> Iterator[str]:
        """Iterate over lines in captured output."""
        return iter(self.captured_output)

    def __next__(self) -> str:
        """Get next line from captured output."""
        return next(iter(self.captured_output))

    @property
    def encoding(self) -> str:
        """Get encoding."""
        return 'utf-8'

    @property
    def errors(self) -> str | None:
        """Get error handling mode."""
        return 'replace'

    @property
    def newlines(self) -> str | tuple[str, ...] | None:
        """Get newline mode."""
        return None

    @property
    def buffer(self) -> Any:
        """Get buffer (not applicable)."""
        raise AttributeError("'CaptureStderr' object has no attribute 'buffer'")

    def getvalue(self) -> str:
        """Get the captured stderr content.
        
        Returns:
            str: The captured stderr output as a string.
        """
        return self.captured_output.getvalue()

    def clear(self) -> None:
        """Clear the captured stderr content (not supported in read mode).
        
        Raises:
            OSError: Always raised as clearing is not supported in read mode.
        """
        raise OSError("not writable")

    def write_to_original(self, text: str) -> None:
        """Write text to the original stderr (bypassing capture).
        
        Args:
            text (str): Text to write to the original stderr.
        """
        if self.original_stderr is not None:
            self.original_stderr.write(text)
            self.original_stderr.flush()
