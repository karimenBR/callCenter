import pandas as pd
import os
from pathlib import Path

def clean_data():
    # Use relative paths that work everywhere
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "raw" / "dataset.csv"
    output_path = project_root / "data" / "processed" / "cleaned_dataset.csv"
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Dataset not found at {input_path}")
        return False
    
    # Load and clean data
    df = pd.read_csv(input_path)
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Clean white spaces
    df["Document"] = df["Document"].str.strip()
    df["Topic_group"] = df["Topic_group"].str.strip()
    
    # Remove any null values
    df = df.dropna()
    
    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Cleaned dataset saved: {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"Categories: {df['Topic_group'].unique()}")
    
    return True

if __name__ == "__main__":
    clean_data()