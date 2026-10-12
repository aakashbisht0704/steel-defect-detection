"""
Environment verification script.
Tests all installed packages and CUDA acceleration.
"""
import sys

def check_env():
    print(">> Python executable:", sys.executable)
    print(">> Python version:", sys.version)

    modules = [
        ("torch", lambda m: f"v{m.__version__} (CUDA available: {m.cuda.is_available()}, Device: {m.cuda.get_device_name(0) if m.cuda.is_available() else 'None'})"),
        ("torchvision", lambda m: f"v{m.__version__}"),
        ("cv2", lambda m: f"v{m.__version__}"),
        ("albumentations", lambda m: f"v{m.__version__}"),
        ("timm", lambda m: f"v{m.__version__}"),
        ("pytorch_grad_cam", lambda m: "Installed"),
        ("sklearn", lambda m: f"v{m.__version__}"),
        ("matplotlib", lambda m: f"v{m.__version__}"),
        ("seaborn", lambda m: f"v{m.__version__}"),
        ("streamlit", lambda m: f"v{m.__version__}"),
        ("pandas", lambda m: f"v{m.__version__}"),
        ("gdown", lambda m: "Installed"),
        ("kaggle", lambda m: "Installed"),
    ]

    all_passed = True
    for mod_name, info_fn in modules:
        try:
            mod = __import__(mod_name)
            info = info_fn(mod)
            print(f" [PASS] {mod_name:<18} -> {info}")
        except Exception as e:
            print(f" [FAIL] {mod_name:<18} -> {e}")
            all_passed = False

    if all_passed:
        print("\n>> All required packages are installed and functional!")
    else:
        print("\n>> Some packages encountered issues.")

if __name__ == "__main__":
    check_env()
