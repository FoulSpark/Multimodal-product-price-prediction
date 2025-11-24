"""
Automated end-to-end pipeline for multimodal price prediction
Runs: Train → Build FAISS → Predict
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 60)
    print(f"🚀 {description}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✅ {description} - COMPLETED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED")
        print(f"Error: {e}")
        return False

def check_prerequisites():
    """Check if required files and packages exist"""
    print("=" * 60)
    print("🔍 CHECKING PREREQUISITES")
    print("=" * 60)
    
    # Check files
    required_files = ['utils.py', 'train.py', 'predict.py', 'build_store.py', 'dataset/train.csv', 'dataset/test.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    
    # Check if we can import required modules
    try:
        import torch
        import transformers
        import lightgbm
        print("✅ Core dependencies installed")
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  No GPU detected - will use CPU (slower)")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def main(args):
    print("\n" + "=" * 60)
    print("🧠 MULTIMODAL PRICE PREDICTION - AUTOMATED PIPELINE")
    print("=" * 60)
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n⚠️  Prerequisites check failed. Please install dependencies first.")
        print("Run: pip install -r requirements.txt")
        return 1
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Train models
    if not args.skip_train:
        train_cmd = [
            sys.executable, 'train.py',
            '--train_csv', args.train_csv,
            '--output_dir', args.model_dir,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--lgb_rounds', str(args.lgb_rounds)
        ]
        
        if not run_command(train_cmd, "TRAINING MODELS"):
            print("\n❌ Training failed. Aborting pipeline.")
            return 1
    else:
        print("\n⏭️  Skipping training (--skip_train flag set)")
    
    # Step 2: Build FAISS index
    if not args.skip_faiss:
        faiss_cmd = [
            sys.executable, 'build_store.py',
            '--embeddings', os.path.join(args.model_dir, 'train_embeddings.pkl'),
            '--output', os.path.join(args.model_dir, 'faiss_index.bin')
        ]
        
        if not run_command(faiss_cmd, "BUILDING FAISS INDEX"):
            print("\n⚠️  FAISS build failed. Continuing without retrieval averaging...")
    else:
        print("\n⏭️  Skipping FAISS index (--skip_faiss flag set)")
    
    # Step 3: Generate predictions
    predict_cmd = [
        sys.executable, 'predict.py',
        '--test_csv', args.test_csv,
        '--model_dir', args.model_dir,
        '--output_dir', args.output_dir,
        '--output_csv', args.output_csv,
        '--weights', str(args.w_mlp), str(args.w_lgb), str(args.w_ret)
    ]
    
    if not run_command(predict_cmd, "GENERATING PREDICTIONS"):
        print("\n❌ Prediction failed. Aborting pipeline.")
        return 1
    
    # Success summary
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("📊 Output Files:")
    print(f"   - Models: {args.model_dir}/")
    print(f"     • train_embeddings.pkl")
    print(f"     • fusion_mlp.pth")
    print(f"     • lgb_model.txt")
    print(f"     • faiss_index.bin")
    print(f"   - Predictions: {os.path.join(args.output_dir, args.output_csv)}")
    print()
    print("📈 Next Steps:")
    print("   1. Check predictions in results/test_out.csv")
    print("   2. Validate format matches sample_test_out.csv")
    print("   3. Submit to competition portal")
    print("   4. Fill out Documentation_template.md")
    print()
    print("=" * 60)
    
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated multimodal price prediction pipeline')
    
    # Data paths
    parser.add_argument('--train_csv', default='dataset/train.csv', help='Training CSV file')
    parser.add_argument('--test_csv', default='dataset/test.csv', help='Test CSV file')
    parser.add_argument('--model_dir', default='models', help='Directory to save models')
    parser.add_argument('--output_dir', default='results', help='Directory to save predictions')
    parser.add_argument('--output_csv', default='test_out.csv', help='Output CSV filename')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lgb_rounds', type=int, default=1000, help='LightGBM boosting rounds')
    
    # Ensemble weights
    parser.add_argument('--w_mlp', type=float, default=0.5, help='MLP weight in ensemble')
    parser.add_argument('--w_lgb', type=float, default=0.3, help='LightGBM weight in ensemble')
    parser.add_argument('--w_ret', type=float, default=0.2, help='Retrieval weight in ensemble')
    
    # Pipeline control
    parser.add_argument('--skip_train', action='store_true', help='Skip training (use existing models)')
    parser.add_argument('--skip_faiss', action='store_true', help='Skip FAISS index building')
    
    args = parser.parse_args()
    
    sys.exit(main(args))
