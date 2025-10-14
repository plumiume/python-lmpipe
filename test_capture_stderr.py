"""Test file for CaptureStderr using multiprocessing.Process.

This test demonstrates that CaptureStderr works correctly in multiprocess environments
and validates the read-only behavior of the implementation.
"""

import sys
import multiprocessing
import time
import os
from pathlib import Path

# Add src to path to import lmpipe
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lmpipe.capture_stderr import CaptureStderr


def worker_function_with_stderr(worker_id: int, shared_list: list):
    """Worker function that writes to stderr and tests CaptureStderr.
    
    Args:
        worker_id (int): Unique identifier for this worker.
        shared_list (list): Shared list to store results.
    """
    try:
        # Test basic stderr writing (this should work normally)
        print(f"Worker {worker_id}: Starting", file=sys.stderr)
        
        # Test CaptureStderr context manager
        with CaptureStderr() as capture:
            print(f"Worker {worker_id}: Inside capture context", file=sys.stderr)
            
            # Try to get captured content (should work)
            captured = capture.getvalue()
            
            # Try to write directly to capture (should raise OSError)
            try:
                capture.write("Direct write test")
                shared_list.append(f"Worker {worker_id}: ERROR - Direct write should have failed")
            except OSError as e:
                shared_list.append(f"Worker {worker_id}: PASS - Direct write correctly failed: {e}")
            
            # Try to write lines directly (should raise OSError)
            try:
                capture.writelines(["Line 1\n", "Line 2\n"])
                shared_list.append(f"Worker {worker_id}: ERROR - writelines should have failed")
            except OSError as e:
                shared_list.append(f"Worker {worker_id}: PASS - writelines correctly failed: {e}")
            
            # Try to truncate (should raise OSError)
            try:
                capture.truncate()
                shared_list.append(f"Worker {worker_id}: ERROR - truncate should have failed")
            except OSError as e:
                shared_list.append(f"Worker {worker_id}: PASS - truncate correctly failed: {e}")
            
            # Try to clear (should raise OSError)
            try:
                capture.clear()
                shared_list.append(f"Worker {worker_id}: ERROR - clear should have failed")
            except OSError as e:
                shared_list.append(f"Worker {worker_id}: PASS - clear correctly failed: {e}")
            
            # Test read operations (should work)
            try:
                readable = capture.readable()
                writable = capture.writable()
                seekable = capture.seekable()
                shared_list.append(f"Worker {worker_id}: PASS - Read operations: readable={readable}, writable={writable}, seekable={seekable}")
            except Exception as e:
                shared_list.append(f"Worker {worker_id}: ERROR - Read operations failed: {e}")
            
            # Test write_to_original (should work)
            try:
                capture.write_to_original(f"Worker {worker_id}: Direct to original stderr\n")
                shared_list.append(f"Worker {worker_id}: PASS - write_to_original worked")
            except Exception as e:
                shared_list.append(f"Worker {worker_id}: ERROR - write_to_original failed: {e}")
            
            # Test fd-based write (simulating C++ direct stderr write)
            try:
                # Get the stderr file descriptor (fd=2)
                stderr_fd = sys.stderr.fileno()
                
                # Write directly to stderr using os.write (simulates C++ fprintf(stderr, ...))
                message = f"Worker {worker_id}: FD write test (C++ simulation)\n".encode('utf-8')
                bytes_written = os.write(stderr_fd, message)
                
                shared_list.append(f"Worker {worker_id}: PASS - FD write worked, wrote {bytes_written} bytes")
            except Exception as e:
                shared_list.append(f"Worker {worker_id}: ERROR - FD write failed: {e}")
        
        shared_list.append(f"Worker {worker_id}: Completed successfully")
        
    except Exception as e:
        shared_list.append(f"Worker {worker_id}: CRITICAL ERROR - {e}")


def test_capture_stderr_functionality():
    """Test the basic functionality of CaptureStderr."""
    print("Testing CaptureStderr basic functionality...")
    
    # Test the convenience function
    try:
        with CaptureStderr() as capture:
            # Test properties
            print(f"Encoding: {capture.encoding}")
            print(f"Errors: {capture.errors}")
            print(f"Closed: {capture.closed}")
            print(f"Readable: {capture.readable()}")
            print(f"Writable: {capture.writable()}")
            print(f"Seekable: {capture.seekable()}")
            
            # Test read operations
            current_pos = capture.tell()
            print(f"Current position: {current_pos}")
            
            # Test getvalue
            captured = capture.getvalue()
            print(f"Currently captured: {repr(captured)}")
            
        print("✓ Basic functionality test passed")
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False
    
    return True


def test_fd_write_capture():
    """Test that fd-based writes (C++ style) are captured correctly."""
    print("Testing fd-based write capture (C++ simulation)...")
    
    try:
        with CaptureStderr() as capture:
            # Test normal Python stderr write
            print("Python stderr write test", file=sys.stderr)
            
            # Test fd-based write (simulating C++ fprintf(stderr, ...))
            stderr_fd = sys.stderr.fileno()
            message = b"C++ style fd write test\n"
            bytes_written = os.write(stderr_fd, message)
            
            # Small delay to ensure the reader thread processes the data
            time.sleep(0.1)
            
            # Check if both writes were captured
            captured = capture.getvalue()
            print(f"Captured content: {repr(captured)}")
            print(f"FD write bytes written: {bytes_written}")
            
            # Verify both messages are in the captured content
            if "Python stderr write test" in captured and "C++ style fd write test" in captured:
                print("✓ Both Python and fd-based writes were captured")
            else:
                print("✗ Some writes were not captured properly")
                return False
                
        print("✓ FD write capture test passed")
        return True
        
    except Exception as e:
        print(f"✗ FD write capture test failed: {e}")
        return False


def test_error_conditions():
    """Test that error conditions are handled correctly."""
    print("Testing error conditions...")
    
    try:
        with CaptureStderr() as capture:
            # Test write operations that should fail
            error_tests = [
                ("write", lambda: capture.write("test")),
                ("writelines", lambda: capture.writelines(["line1\n", "line2\n"])),
                ("truncate", lambda: capture.truncate()),
                ("clear", lambda: capture.clear()),
            ]
            
            for test_name, test_func in error_tests:
                try:
                    test_func()
                    print(f"✗ {test_name} should have raised OSError")
                    return False
                except OSError:
                    print(f"✓ {test_name} correctly raised OSError")
                except Exception as e:
                    print(f"✗ {test_name} raised unexpected error: {e}")
                    return False
        
        print("✓ Error conditions test passed")
        return True
        
    except Exception as e:
        print(f"✗ Error conditions test failed: {e}")
        return False


def main():
    """Main test function using multiprocessing."""
    print("=== CaptureStderr Multiprocessing Test ===\n")
    
    # Test basic functionality first
    if not test_capture_stderr_functionality():
        print("Basic functionality test failed, aborting multiprocessing test")
        return
    
    print()
    
    # Test fd-based write capture
    if not test_fd_write_capture():
        print("FD write capture test failed, aborting multiprocessing test")
        return
    
    print()
    
    # Test error conditions
    if not test_error_conditions():
        print("Error conditions test failed, aborting multiprocessing test")
        return
    
    print()
    
    # Multiprocessing test
    print("Starting multiprocessing test...")
    
    # Use Manager to share data between processes
    with multiprocessing.Manager() as manager:
        shared_results = manager.list()
        
        # Create and start multiple processes
        processes = []
        num_processes = 3
        
        for i in range(num_processes):
            p = multiprocessing.Process(
                target=worker_function_with_stderr,
                args=(i, shared_results)
            )
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=10)  # 10 second timeout
            if p.is_alive():
                print(f"Process {p.pid} timed out, terminating...")
                p.terminate()
                p.join()
        
        # Print results
        print("\n=== Multiprocessing Test Results ===")
        for result in shared_results:
            print(result)
        
        # Count passes and failures
        results_list = list(shared_results)
        passes = len([r for r in results_list if "PASS" in r])
        errors = len([r for r in results_list if "ERROR" in r])
        
        print(f"\n=== Summary ===")
        print(f"Total processes: {num_processes}")
        print(f"Passes: {passes}")
        print(f"Errors: {errors}")
        
        if errors == 0:
            print("✓ All multiprocessing tests passed!")
        else:
            print("✗ Some tests failed")


if __name__ == "__main__":
    # Ensure multiprocessing works on Windows
    multiprocessing.set_start_method('spawn', force=True)
    main()