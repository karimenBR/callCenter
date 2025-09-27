import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_data():
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "processed" / "cleaned_dataset.csv"
    
    if not input_path.exists():
        print(f"Error: Cleaned dataset not found at {input_path}")
        print("Please run clean_data.py first")
        return False
    
    # Load cleaned data
    df = pd.read_csv(input_path)
    print(f"Loading dataset: {df.shape}")
    
    # Split: 51% train, 15% val, 34% test
    df_train, df_temp = train_test_split(
        df, test_size=0.49, random_state=42, 
        stratify=df["Topic_group"]
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.6939, random_state=42, 
        stratify=df_temp["Topic_group"]
    )
    
    # Save splits
    processed_dir = project_root / "data" / "processed"
    df_train.to_csv(processed_dir / "train.csv", index=False)
    df_val.to_csv(processed_dir / "val.csv", index=False)
    df_test.to_csv(processed_dir / "test.csv", index=False)
    
    print("\n✅ Dataset split completed:")
    print(f"Train: {len(df_train)} rows ({len(df_train)/len(df)*100:.1f}%)")
    print(f"Validation: {len(df_val)} rows ({len(df_val)/len(df)*100:.1f}%)")
    print(f"Test: {len(df_test)} rows ({len(df_test)/len(df)*100:.1f}%)")
    
    # Verify class distribution
    print("\n📊 Class distribution:")
    for split_name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        print(f"{split_name}: {split_df['Topic_group'].value_counts().to_dict()}")
    
    return True

if __name__ == "__main__":
    split_data()