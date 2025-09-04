import os
import json
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from pipeline import reader, preprocessor, dataset
from utils import drawAUC_TwoClass


def parse_args():
    parser = argparse.ArgumentParser(description="Train Random Forest on data")
    parser.add_argument("--feature_path", default='Data/samodra/feature/', type=str)
    parser.add_argument("--label_path", default='Data/samodra/label/label2.tif', type=str)
    parser.add_argument("--output_dir", default='Hasil_RF_samodra/', type=str)
    parser.add_argument("--window_size", default=15, type=int)
    parser.add_argument("--n_estimators", default=200, type=int)
    parser.add_argument("--max_depth", default=None, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # --- data validation ---
    reader.validate_consistency(args.feature_path, args.label_path)

    # --- load and processing data ---
    feature_files = sorted([f for f in os.listdir(args.feature_path) if f.lower().endswith('.tif')])

    padded_features = []
    n = args.window_size // 2
    
    # processing features
    for feature_name in feature_files:
        img, _, _ = reader.read_data_from_tif(os.path.join(args.feature_path, feature_name))
        norm_img, _, _ = preprocessor.normalize_min_max(img)
        padded_img = preprocessor.apply_padding(norm_img, n, pad_value=0)
        padded_features.append(padded_img)
        
    feature_block = np.array(padded_features)
    print(f"Feature block created successfully: {feature_block.shape}")
    
    # processing label
    label_img, _, _ = reader.read_data_from_tif(args.label_path)
    padded_label = preprocessor.apply_padding(label_img, n, pad_value=0.1)
    
    # --- create dataset ---
    train_x, train_y, val_x, val_y = dataset.get_CNN_data(
        feature_block, padded_label, args.window_size
    )
    print(f"Dataset created: {train_x.shape[0]} train data, {val_x.shape[0]} val data.")

    # reshape ke format [n_samples, n_features]
    train_x = train_x.reshape(train_x.shape[0], -1)
    val_x = val_x.reshape(val_x.shape[0], -1)

    # --- Train Random Forest ---
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=42
    )

    model.fit(train_x, train_y)

    # --- Evaluate ---
    train_preds = model.predict(train_x)
    val_preds = model.predict(val_x)

    train_acc = accuracy_score(train_y, train_preds)
    val_acc = accuracy_score(val_y, val_preds)

    train_probs = model.predict_proba(train_x)[:, 1]
    val_probs = model.predict_proba(val_x)[:, 1]

    train_auc = roc_auc_score(train_y, train_probs)
    val_auc = roc_auc_score(val_y, val_probs)

    print(f"Train Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
    print(f"Val Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

    # --- Save model and results ---
    os.makedirs(args.output_dir, exist_ok=True)
    import joblib
    joblib.dump(model, os.path.join(args.output_dir, "rf_model.pkl"))

    drawAUC_TwoClass(val_y, val_probs, os.path.join(args.output_dir, 'val_AUC.png'))
    drawAUC_TwoClass(train_y, train_probs, os.path.join(args.output_dir, 'train_AUC.png'))

    record = {
        "train_acc": train_acc,
        "train_auc": train_auc,
        "val_acc": val_acc,
        "val_auc": val_auc
    }
    with open(os.path.join(args.output_dir, 'record.json'), 'w') as f:
        json.dump(record, f, indent=4)

    print("Training selesai dengan Random Forest.")


if __name__ == '__main__':
    main()
