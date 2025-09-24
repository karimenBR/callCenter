import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv(r"C:\Users\karim\callCenter\data\raw\dataset.csv")
df_train,df_temp=train_test_split(df,test_size=0.49, random_state=42, stratify=df["Topic_group"])
df_val,df_test=train_test_split(df_temp, test_size=0.6939, random_state=42, stratify=df_temp["Topic_group"]) #hatina 0.6939 bach nalkaw 15% w 34%

df_train.to_csv(r"C:\Users\karim\callCenter\data\processed\train.csv", index=False)
df_val.to_csv(r"C:\Users\karim\callCenter\data\processed/val.csv", index=False)
df_test.to_csv(r"C:\Users\karim\callCenter\data\processed/test.csv", index=False)

print("\ntadaaa Dataset split completed:")
print(f"Train: {len(df_train)} rows")
print(f"Validation: {len(df_val)} rows")
print(f"Test: {len(df_test)} rows")