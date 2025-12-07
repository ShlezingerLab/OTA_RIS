"""
Quick demo script to test if the reproduce_fig5.py script works.
This runs a minimal test with just 1 configuration, 1 restart, and 3 epochs.
"""

import subprocess
import sys
import os

def main():
    print("="*70)
    print("Running DEMO test of reproduce_fig5.py")
    print("="*70)
    print("\nThis will test:")
    print("  - 1 configuration (No Metasurface)")
    print("  - 1 training restart")
    print("  - 3 epochs")
    print("  - 100 training samples")
    print("\nThis should complete in a few minutes...")
    print("="*70)

    # Run the reproduce script with minimal demo parameters
    script_path = os.path.join(os.path.dirname(__file__), "reproduce_fig5.py")

    cmd = [
        sys.executable,
        script_path,
        "--demo"
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*70)
        print("DEMO TEST COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nYou can now run the full script with:")
        print("  python reproduce_fig5.py --quick-test")
        print("\nOr customize parameters:")
        print("  python reproduce_fig5.py --num-restarts 5 --num-epochs 50")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Demo test failed with exit code {e.returncode}")
        return 1
    except KeyboardInterrupt:
        print("\n\nDemo test interrupted by user")
        return 1

if __name__ == '__main__':
    sys.exit(main())
