"""
Quick verification script to check if all files are connected properly
"""
import sys
import os

def check_files():
    """Check if all required files exist"""
    print("=" * 60)
    print("🔍 CHECKING FILE STRUCTURE")
    print("=" * 60)
    
    required_files = {
        'utils.py': 'Core utilities (CLIP, Text encoder, OCR)',
        'train.py': 'Training script',
        'predict.py': 'Prediction script',
        'build_store.py': 'FAISS index builder',
        'requirements.txt': 'Dependencies',
        'dataset/train.csv': 'Training data (75K samples)',
        'dataset/test.csv': 'Test data (75K samples)',
        'dataset/sample_test_out.csv': 'Expected output format'
    }
    
    all_good = True
    for file, desc in required_files.items():
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"{status} {file:30s} - {desc}")
        if not exists:
            all_good = False
    
    print()
    return all_good

def check_imports():
    """Check if all imports work"""
    print("=" * 60)
    print("🔍 CHECKING IMPORTS")
    print("=" * 60)
    
    imports_to_check = [
        ('torch', 'PyTorch'),
        ('transformers', 'Hugging Face Transformers'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('lightgbm', 'LightGBM'),
        ('PIL', 'Pillow'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
    ]
    
    optional_imports = [
        ('paddleocr', 'PaddleOCR (optional but recommended)'),
        ('pytesseract', 'Pytesseract (optional fallback)'),
        ('faiss', 'FAISS (optional but recommended)'),
    ]
    
    all_good = True
    for module, name in imports_to_check:
        try:
            __import__(module)
            print(f"✅ {name:40s} - Installed")
        except ImportError:
            print(f"❌ {name:40s} - MISSING (required)")
            all_good = False
    
    print("\nOptional packages:")
    for module, name in optional_imports:
        try:
            __import__(module)
            print(f"✅ {name:40s} - Installed")
        except ImportError:
            print(f"⚠️  {name:40s} - Not installed (optional)")
    
    print()
    return all_good

def check_custom_imports():
    """Check if custom modules import correctly"""
    print("=" * 60)
    print("🔍 CHECKING CUSTOM MODULES")
    print("=" * 60)
    
    try:
        from utils import CLIPImageEncoder, TextEncoder, OCRExtractor, download_image, build_and_save_feature_store
        print("✅ utils.py - All classes import successfully")
        print("   - CLIPImageEncoder ✅")
        print("   - TextEncoder ✅")
        print("   - OCRExtractor ✅")
        print("   - download_image ✅")
        print("   - build_and_save_feature_store ✅")
    except Exception as e:
        print(f"❌ utils.py - Import failed: {e}")
        return False
    
    try:
        from train import FusionRegressor, EmbeddingDataset
        print("✅ train.py - All classes import successfully")
        print("   - FusionRegressor ✅")
        print("   - EmbeddingDataset ✅")
    except Exception as e:
        print(f"❌ train.py - Import failed: {e}")
        return False
    
    print()
    return True

def check_csv_format():
    """Check CSV format matches expectations"""
    print("=" * 60)
    print("🔍 CHECKING CSV FORMAT")
    print("=" * 60)
    
    try:
        import pandas as pd
        
        # Check train.csv
        train = pd.read_csv('dataset/train.csv', nrows=5)
        print("✅ train.csv loaded successfully")
        print(f"   Columns: {list(train.columns)}")
        
        expected_cols = ['sample_id', 'catalog_content', 'image_link', 'price']
        missing = set(expected_cols) - set(train.columns)
        if missing:
            print(f"   ⚠️  Missing columns: {missing}")
        else:
            print(f"   ✅ All expected columns present")
        
        print(f"   Sample data:")
        print(f"   - sample_id: {train['sample_id'].iloc[0]}")
        print(f"   - catalog_content: {train['catalog_content'].iloc[0][:50]}...")
        print(f"   - image_link: {train['image_link'].iloc[0][:50]}...")
        print(f"   - price: {train['price'].iloc[0]}")
        
        # Check test.csv
        test = pd.read_csv('dataset/test.csv', nrows=5)
        print("\n✅ test.csv loaded successfully")
        print(f"   Columns: {list(test.columns)}")
        
        print()
        return True
    except Exception as e:
        print(f"❌ CSV check failed: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("🚀 MULTIMODAL PRICE PREDICTION - SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    results = []
    
    # Check files
    results.append(("File Structure", check_files()))
    
    # Check imports
    results.append(("Required Imports", check_imports()))
    
    # Check custom imports
    results.append(("Custom Modules", check_custom_imports()))
    
    # Check CSV format
    results.append(("CSV Format", check_csv_format()))
    
    # Summary
    print("=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10s} - {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL CHECKS PASSED! You're ready to train.")
        print()
        print("Next steps:")
        print("1. python train.py --train_csv dataset/train.csv --output_dir models")
        print("2. python build_store.py --embeddings models/train_embeddings.pkl --output models/faiss_index.bin")
        print("3. python predict.py --test_csv dataset/test.csv --model_dir models --output_dir results")
    else:
        print("⚠️  SOME CHECKS FAILED. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Check file paths and names")
        print("- Ensure dataset files are in dataset/ folder")
    
    print("=" * 60)
    print()
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
