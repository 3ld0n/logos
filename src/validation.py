# Script for Step 7: Validation and Analysis
# This script loads extracted entities, creates a DataFrame, counts occurrences,
# and generates basic visualizations.
# Decisions:
# - Input File: Using 'lexicon_entities.json'; alternative: Switch to other outputs like 'custom_trained_entities.json'.
# - Visualization: Simple stripplot; option: Use barplot for counts or heatmap for book distribution.
# - Metrics: Basic Counter; alternative: Add scikit-learn for precision/recall if gold data available.

import json
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def load_entities(file_path='lexicon_entities.json'):
    """Load extracted entities from JSON."""
    with open(file_path, 'r') as f:
        entities = json.load(f)
    return entities

def analyze_entities(entities):
    """Analyze with pandas and Counter."""
    df = pd.DataFrame(entities)
    entity_counts = Counter(df['entity'])
    print("Top 10 entities:", entity_counts.most_common(10))
    # Decision: Save to CSV for external validation; option: Cross-check with API here.
    df.to_csv('entities_df.csv', index=False)
    return df

def visualize_entities(df):
    """Visualize with seaborn."""
    sns.stripplot(x='verse', y='entity', data=df.head(1000))  # Limit for performance
    plt.title("Entity Distribution in First 1000 Verses")
    plt.savefig('entity_visual.png')
    plt.show()

if __name__ == "__main__":
    entities = load_entities()
    df = analyze_entities(entities)
    visualize_entities(df)
    print("Analysis complete. Check 'entities_df.csv' and 'entity_visual.png'.")