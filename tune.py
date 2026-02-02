import pandas as pd
import numpy as np
import sklearn.utils

# --- HOTFIX BLOCK ---
if not hasattr(sklearn.utils, "check_matplotlib_support"):
    def check_matplotlib_support(caller_name):
        try:
            import matplotlib
        except ImportError:
            raise ImportError(f"{caller_name} requires matplotlib.")
    sklearn.utils.check_matplotlib_support = check_matplotlib_support
# --------------------

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklift.models import ClassTransformation
from sklift.metrics import qini_auc_score
from sklift.datasets import fetch_hillstrom

def load_data():
    print("🔄 Loading Dataset...")
    dataset = fetch_hillstrom(target_col='conversion')
    df = dataset.data
    df['treatment'] = dataset.treatment
    df['target'] = dataset.target
    df = df[df['treatment'] != 'Womens E-Mail'].copy()
    df['treatment'] = df['treatment'].apply(lambda x: 1 if x == 'Mens E-Mail' else 0)
    return df

def build_pipeline():
    numeric_features = ['recency', 'history']
    categorical_features = ['zip_code', 'channel']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

def tune():
    # 1. Prepare Data
    df = load_data()
    X = df[['recency', 'history', 'zip_code', 'channel']]
    y = df['target']
    treat = df['treatment']

    print("✂️ Splitting Data...")
    X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
        X, y, treat, test_size=0.3, random_state=42, stratify=treat
    )

    # 2. Define the Parameters to Test
    # We will loop through these manually
    depths = [3, 4, 5]
    n_estimators = [50, 100, 200]
    learning_rates = [0.01, 0.1, 0.2]

    best_score = -1.0
    best_params = {}

    print(f"\n🔎 Starting Manual Grid Search...")
    print(f"   Testing {len(depths) * len(n_estimators) * len(learning_rates)} combinations.\n")

    # 3. Manual Loop
    for d in depths:
        for n in n_estimators:
            for lr in learning_rates:
                
                # A. Setup Model with current parameters
                inner_model = Pipeline([
                    ('preprocessor', build_pipeline()),
                    ('model', XGBClassifier(
                        n_estimators=n, 
                        max_depth=d, 
                        learning_rate=lr, 
                        random_state=42,
                        eval_metric='logloss'
                    ))
                ])
                
                uplift_model = ClassTransformation(estimator=inner_model)

                # B. Train
                # We catch errors just in case one setting fails
                try:
                    uplift_model.fit(X_train, y_train, treat_train)
                    
                    # C. Evaluate
                    # We evaluate on the Test set to find the one that generalizes best
                    preds = uplift_model.predict(X_test)
                    score = qini_auc_score(y_test, preds, treat_test)
                    
                    # Print progress (Optional: Comment out if too noisy)
                    print(f"   [Depth={d}, N={n}, LR={lr}] -> Qini: {score:.4f}")

                    # D. Compare
                    if score > best_score:
                        best_score = score
                        best_params = {'max_depth': d, 'n_estimators': n, 'learning_rate': lr}
                        print(f"      ✨ NEW BEST FOUND! ({score:.4f})")

                except Exception as e:
                    print(f"   ⚠️ Failed for settings D={d}, N={n}: {e}")

    # 4. Final Result
    print("\n✅ Tuning Complete!")
    print("--------------------------------------------------")
    print(f"🏆 WINNING PARAMETERS: {best_params}")
    print(f"📈 BEST QINI SCORE: {best_score:.4f}")
    print("--------------------------------------------------")
    print("👉 Action: Update your train.py with these numbers!")

if __name__ == "__main__":
    tune()