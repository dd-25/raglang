import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now we can import from src
from src.main import main

if __name__ == "__main__":
    main()
