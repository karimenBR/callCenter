import pandas as pd

df = pd.read_csv(r"C:\Users\karim\callCenter\data\raw\dataset.csv")
print(df.columns)

#cleaning white spaces
df["Document"]=df["Document"].str.strip()
df["Topic_group"]=df["Topic_group"].str.strip()

#output
print(df.head())