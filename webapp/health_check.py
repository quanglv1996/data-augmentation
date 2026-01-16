"""
Health check script for Data Augmentation Web App
Run this to verify installation
"""

import sys
import os

def check_python_version():
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.9+)")
        return False

def check_dependencies():
    print("\n📦 Checking dependencies...")
    required_packages = [
        'flask',
        'cv2',
        'numpy',
        'PIL'
    ]
    
    all_ok = True
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"   ✅ opencv-python")
            elif package == 'PIL':
                import PIL
                print(f"   ✅ Pillow")
            else:
                __import__(package)
                print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} not found")
            all_ok = False
    
    return all_ok

def check_augmentation_modules():
    print("\n🎨 Checking augmentation modules...")
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    modules = [
        'augmentations.brightness',
        'augmentations.contrast',
        'augmentations.horizontal_flip',
        'augmentations.rotate',
        'utils.utils'
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module} - {str(e)}")
            all_ok = False
    
    return all_ok

def check_directories():
    print("\n📁 Checking directories...")
    
    required_dirs = [
        'templates',
        'static'
    ]
    
    all_ok = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ not found")
            all_ok = False
    
    return all_ok

def main():
    print("="*50)
    print("Data Augmentation Web App - Health Check")
    print("="*50)
    
    results = []
    
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Augmentation Modules", check_augmentation_modules()))
    results.append(("Directories", check_directories()))
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False
    
    print("="*50)
    
    if all_passed:
        print("\n🎉 All checks passed! Ready to run the application.")
        print("\nTo start:")
        print("  1. Using Docker: docker-compose up -d --build")
        print("  2. Direct Python: python app.py")
        print("\nAccess: http://localhost:222")
    else:
        print("\n⚠️  Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    print()

if __name__ == "__main__":
    main()
