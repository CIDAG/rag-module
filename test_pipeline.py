import subprocess
import sys
from pathlib import Path

def run_sequential_tests() -> None:
    """
    Discover and execute all test files sequentially using pytest.
    
    This script automatically runs each test file matching 'test_*.py' inside
    the 'tests' directory and generates an isolated HTML report for each one
    to avoid overwriting results and help guide code corrections.
    """
    # Define the paths using pathlib.Path as per guidelines
    root_dir = Path(__file__).parent
    tests_dir = root_dir / "tests"
    reports_dir = root_dir / "reports"

    if not tests_dir.exists():
        print(f"Error: 'tests/' directory not found at {tests_dir}")
        sys.exit(1)

    # Automatically create the reports directory if it doesn't exist
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Find and sort all test files to ensure sequential, deterministic execution
    test_files = sorted(tests_dir.glob("test_*.py"))

    if not test_files:
        print("No test files found matching 'test_*.py' inside the tests directory.")
        return

    print(f"Starting sequential test execution for {len(test_files)} files...\n")

    for test_file in test_files:
        # Generate a unique report name per test file to prevent overwriting
        report_name = f"tests_report_{test_file.stem}.html"
        report_path = reports_dir / report_name
        relative_path = test_file.relative_to(root_dir)
        
        print(f"==================================================================")
        print(f"Running: {relative_path}")
        print(f"==================================================================")

        # Build the pytest command following the user's requested pattern
        command = [
            "pytest",
            str(relative_path),
            f"--html={report_path}",
            "--self-contained-html"
        ]

        try:
            # Execute the command and stream output directly to the console
            result = subprocess.run(command, capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"\nSTATUS: SUCCESS - {test_file.name} passed completely.")
            else:
                print(f"\nSTATUS: FAILURE - {test_file.name} failed (Exit code: {result.returncode}).")
            
            print(f"Report successfully generated: {report_name}\n")
            
        except Exception as e:
            print(f"An unexpected error occurred while executing {test_file.name}: {e}\n")

if __name__ == "__main__":
    run_sequential_tests()