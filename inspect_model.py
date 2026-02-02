import joblib
import pandas as pd

# 1. Load the frozen model
print("🧊 Unfreezing the model...")
uplift_model = joblib.load('uplift_model.pkl')

try:
    # 2. Extract the XGBoost Booster
    # Structure: ClassTransformation -> Pipeline -> XGBClassifier -> Booster
    pipeline = uplift_model.estimator
    xgboost_model = pipeline.named_steps['model']
    
    # Get the actual "Booster" object (the core brain)
    booster = xgboost_model.get_booster()

    # 3. Extract the Feature Names
    # We need to know what "f0", "f1" actually mean (e.g., "recency")
    # The pipeline's preprocessor has the feature names
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # 4. Get the Feature Importance (What matters most?)
    # This tells us which columns the model uses most often to make decisions
    importance = booster.get_score(importance_type='weight')
    
    # Map "f0" to "num__recency"
    mapped_importance = {feature_names[int(k[1:])]: v for k, v in importance.items()}
    
    # Convert to DataFrame for nice printing
    df_importance = pd.DataFrame(list(mapped_importance.items()), columns=['Feature', 'Weight'])
    df_importance = df_importance.sort_values(by='Weight', ascending=False)
    
    print("\n🧠 WHAT YOUR MODEL LEARNED (Feature Importance):")
    print("-------------------------------------------------")
    print(df_importance)
    print("-------------------------------------------------")
    print("Interpretation: Higher 'Weight' means this feature is used more often in IF/THEN rules.")

    # 5. Print the first few "Rules" as text
    # This shows the actual logic of the first tree
    print("\n📜 ACTUAL DECISION RULES (First Tree as Text):")
    # limits to just the first tree (index 0)
    print(booster.get_dump()[0]) 

except Exception as e:
    print(f"❌ Error inspecting model: {e}")