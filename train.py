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
import joblib
import matplotlib.pyplot as plt

# Import the viz tools
from sklift.models import ClassTransformation
from sklift.metrics import qini_auc_score
from sklift.viz import plot_qini_curve, plot_uplift_by_percentile  # <--- Added this import
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

def train():
    df = load_data()
    
    X = df[['recency', 'history', 'zip_code', 'channel']]
    y = df['target']
    treat = df['treatment']

    print(f"📊 Data Shape: {X.shape}")

    # Split Data
    X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
        X, y, treat, test_size=0.3, random_state=42, stratify=treat
    )

    print("🤖 Training Class Transformation Model (Optimized)...")
    
    estimator = Pipeline([
        ('preprocessor', build_pipeline()),
        ('model', XGBClassifier(
            n_estimators=100, 
            max_depth=3, 
            learning_rate=0.2, 
            random_state=42
        ))
    ])

    uplift_model = ClassTransformation(estimator=estimator)
    uplift_model.fit(X_train, y_train, treat_train)
    print("✅ Training Complete.")

    # Evaluate
    uplift_preds = uplift_model.predict(X_test)
    qini = qini_auc_score(y_test, uplift_preds, treat_test)
    print(f"\n📈 Final Qini AUC Score: {qini:.4f}")

    # --- MANUAL PLOTTING SECTION (Guaranteed to work) ---
    print("🎨 Generating Plots...")
    
    # 1. Plot Qini Curve (This usually works)
    try:
        plot_qini_curve(y_test, uplift_preds, treat_test, perfect=False)
        plt.title(f'Optimized Qini Curve (AUC={qini:.4f})')
        plt.savefig('qini_curve.png')
        print("   -> Saved 'qini_curve.png'")
    except Exception as e:
        print(f"   ⚠️ Qini Plot error: {e}")

    # 2. MANUAL "Uplift by Percentile" (Replaces the broken function)
    try:
        # A. Create a DataFrame with all the results
        results = pd.DataFrame({
            'y_true': y_test.values,
            'uplift_score': uplift_preds,
            'treatment': treat_test.values
        })
        
        # B. Sort by Score (High to Low) and split into 10 groups (Deciles)
        results['decile'] = pd.qcut(results['uplift_score'], 10, labels=False, duplicates='drop')
        results['decile'] = 9 - results['decile'] # Reverse so 0 is Top 10%
        
        # C. Calculate Uplift for each Group
        # Uplift = (Conversion_Treated - Conversion_Control)
        uplift_by_decile = results.groupby('decile').apply(
            lambda x: x[x['treatment'] == 1]['y_true'].mean() - x[x['treatment'] == 0]['y_true'].mean()
        )
        
        # D. Plot it
        plt.figure(figsize=(10, 6))
        uplift_by_decile.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Uplift by Decile (Manual Calculation)')
        plt.xlabel('Decile (0 = Top 10% Best Customers)')
        plt.ylabel('Uplift (Difference in Conversion Rate)')
        plt.axhline(0, color='grey', linestyle='--', linewidth=1)
        plt.savefig('uplift_by_percentile.png')
        print("   -> Saved 'uplift_by_percentile.png'")

    except Exception as e:
        print(f"   ⚠️ Manual Plotting error: {e}")
    # ----------------------------------------------------

    joblib.dump(uplift_model, 'uplift_model.pkl')
    print("\n✅ All Artifacts Saved.")

if __name__ == "__main__":
    train()