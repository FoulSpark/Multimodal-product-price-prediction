"""
Script to test predictions on different datasets
"""
import argparse
import subprocess
import sys
import os
import pandas as pd

def predict_on_dataset(test_csv, output_csv='predictions.csv', model_dir='models'):
    """
    Generate predictions for any dataset
    
    Args:
        test_csv: Path to test CSV file
        output_csv: Where to save predictions
        model_dir: Directory containing trained models
    """
    
    print("=" * 60)
    print(f"🔮 GENERATING PREDICTIONS")
    print("=" * 60)
    print(f"Input: {test_csv}")
    print(f"Output: {output_csv}")
    print(f"Models: {model_dir}")
    print()
    
    # Check if test file exists
    if not os.path.exists(test_csv):
        print(f"❌ Error: Test file not found: {test_csv}")
        return False
    
    # Check if models exist
    required_models = [
        os.path.join(model_dir, 'train_embeddings.pkl'),
        os.path.join(model_dir, 'fusion_mlp.pth'),
        os.path.join(model_dir, 'lgb_model.txt')
    ]
    
    missing_models = [m for m in required_models if not os.path.exists(m)]
    if missing_models:
        print("❌ Error: Missing trained models:")
        for m in missing_models:
            print(f"   - {m}")
        print("\nRun training first: python train.py")
        return False
    
    # Load and show dataset info
    df = pd.read_csv(test_csv)
    print(f"✅ Dataset loaded: {len(df)} samples")
    print(f"   Columns: {list(df.columns)}")
    print()
    
    # Run prediction
    cmd = [
        sys.executable, 'predict.py',
        '--test_csv', test_csv,
        '--model_dir', model_dir,
        '--output_dir', os.path.dirname(output_csv) or '.',
        '--output_csv', os.path.basename(output_csv)
    ]
    
    print("Running prediction...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print()
        print("=" * 60)
        print("✅ PREDICTION COMPLETED")
        print("=" * 60)
        
        # Show output info
        if os.path.exists(output_csv):
            out_df = pd.read_csv(output_csv)
            print(f"Output saved: {output_csv}")
            print(f"Predictions: {len(out_df)} samples")
            print()
            print("First 10 predictions:")
            print(out_df.head(10))
            print()
            print(f"Price statistics:")
            print(f"  Min: ${out_df['price'].min():.2f}")
            print(f"  Max: ${out_df['price'].max():.2f}")
            print(f"  Mean: ${out_df['price'].mean():.2f}")
            print(f"  Median: ${out_df['price'].median():.2f}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("❌ PREDICTION FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test predictions on different datasets')
    parser.add_argument('--test_csv', required=True, help='Path to test CSV file')
    parser.add_argument('--output_csv', default='predictions.csv', help='Output CSV filename')
    parser.add_argument('--model_dir', default='models', help='Directory with trained models')
    
    args = parser.parse_args()
    
    success = predict_on_dataset(args.test_csv, args.output_csv, args.model_dir)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
