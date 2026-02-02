import pandas as pd
import numpy as np

# Create 10 fake customers with different behaviors
data = {
    # Recency: Months since last purchase (Lower is better)
    'recency': [1, 12, 6, 2, 10, 1, 5, 8, 3, 11],
    
    # History: Total money spent (Higher is usually better)
    'history': [200.50, 29.99, 100.00, 500.00, 45.00, 150.00, 80.00, 30.00, 300.00, 25.00],
    
    # Zip Code: Categorical (Must match training data)
    'zip_code': ['Urban', 'Rural', 'Suburban', 'Urban', 'Rural', 'Urban', 'Suburban', 'Urban', 'Rural', 'Suburban'],
    
    # Channel: How they usually buy (Phone or Web)
    'channel': ['Web', 'Phone', 'Web', 'Web', 'Phone', 'Phone', 'Web', 'Web', 'Phone', 'Web']
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('new_customers.csv', index=False)
print("✅ Created 'new_customers.csv' successfully!")