"""
Quick script to validate output format against sample
"""
import pandas as pd

def validate_output(output_file='results/test_out.csv', sample_file='dataset/sample_test_out.csv'):
    """Validate output format matches sample"""
    
    print("=" * 60)
    print("📊 OUTPUT VALIDATION")
    print("=" * 60)
    
    # Load sample format
    try:
        sample = pd.read_csv(sample_file)
        print(f"\n✅ Sample file loaded: {sample_file}")
        print(f"   Rows: {len(sample)}")
        print(f"   Columns: {list(sample.columns)}")
        print(f"\n   First 5 rows:")
        print(sample.head())
    except FileNotFoundError:
        print(f"❌ Sample file not found: {sample_file}")
        return False
    
    # Load your output
    try:
        output = pd.read_csv(output_file)
        print(f"\n✅ Your output loaded: {output_file}")
        print(f"   Rows: {len(output)}")
        print(f"   Columns: {list(output.columns)}")
        print(f"\n   First 5 rows:")
        print(output.head())
    except FileNotFoundError:
        print(f"\n⚠️  Output file not found: {output_file}")
        print("   Run the pipeline first: python run_pipeline.py")
        return False
    
    # Validation checks
    print("\n" + "=" * 60)
    print("🔍 VALIDATION CHECKS")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: Column names
    if list(output.columns) == list(sample.columns):
        print("✅ Check 1: Column names match")
        checks_passed += 1
    else:
        print(f"❌ Check 1: Column mismatch")
        print(f"   Expected: {list(sample.columns)}")
        print(f"   Got: {list(output.columns)}")
    
    # Check 2: Column types
    if 'sample_id' in output.columns and 'price' in output.columns:
        print("✅ Check 2: Required columns present (sample_id, price)")
        checks_passed += 1
    else:
        print("❌ Check 2: Missing required columns")
    
    # Check 3: No missing values
    missing = output.isnull().sum().sum()
    if missing == 0:
        print(f"✅ Check 3: No missing values")
        checks_passed += 1
    else:
        print(f"❌ Check 3: Found {missing} missing values")
    
    # Check 4: All prices are positive
    if (output['price'] > 0).all():
        print("✅ Check 4: All prices are positive")
        checks_passed += 1
    else:
        negative_count = (output['price'] <= 0).sum()
        print(f"❌ Check 4: Found {negative_count} non-positive prices")
    
    # Check 5: Price range reasonable
    price_min = output['price'].min()
    price_max = output['price'].max()
    price_mean = output['price'].mean()
    
    print(f"✅ Check 5: Price statistics")
    print(f"   Min: ${price_min:.2f}")
    print(f"   Max: ${price_max:.2f}")
    print(f"   Mean: ${price_mean:.2f}")
    checks_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📈 VALIDATION SUMMARY: {checks_passed}/{total_checks} checks passed")
    print("=" * 60)
    
    if checks_passed == total_checks:
        print("🎉 All checks passed! Output is ready for submission.")
        return True
    else:
        print("⚠️  Some checks failed. Please review and fix.")
        return False

if __name__ == '__main__':
    import sys
    
    # Check if custom output file provided
    output_file = sys.argv[1] if len(sys.argv) > 1 else 'results/test_out.csv'
    
    validate_output(output_file)
