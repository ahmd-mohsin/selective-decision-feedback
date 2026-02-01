import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("Project root:", project_root)
print("Python path:", sys.path[:3])
print("\nChecking transmission_system directory:")

ts_dir = project_root / "transmission_system"
print(f"Directory exists: {ts_dir.exists()}")

if ts_dir.exists():
    print("\nFiles in transmission_system/:")
    for item in sorted(ts_dir.iterdir()):
        print(f"  {item.name}")
    
    print("\nTrying imports one by one:")
    
    try:
        from transmission_system import config
        print("✓ config imported successfully")
    except Exception as e:
        print(f"✗ config import failed: {e}")
    
    try:
        from transmission_system import constellation
        print("✓ constellation imported successfully")
    except Exception as e:
        print(f"✗ constellation import failed: {e}")
    
    try:
        from transmission_system import modulator
        print("✓ modulator imported successfully")
    except Exception as e:
        print(f"✗ modulator import failed: {e}")
    
    try:
        from transmission_system import channel
        print("✓ channel imported successfully")
    except Exception as e:
        print(f"✗ channel import failed: {e}")
    
    try:
        from transmission_system import receiver_frontend
        print("✓ receiver_frontend imported successfully")
    except Exception as e:
        print(f"✗ receiver_frontend import failed: {e}")
    
    try:
        from transmission_system import dataset_generator
        print("✓ dataset_generator imported successfully")
    except Exception as e:
        print(f"✗ dataset_generator import failed: {e}")