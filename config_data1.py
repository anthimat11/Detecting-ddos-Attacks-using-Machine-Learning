import pandas as pd

data = pd.read_csv('oversampled_dataset.csv')

# Sample data with an 'IP' column containing IP addresses in the format 10.0.0.X
data_src = data["src"]
data_src = data["dst"]
data_protocol = data["Protocol"]
df = pd.DataFrame(data)

# Extract the last segment of the IP address and replace the 'IP' column with numeric values
df['src'] = df['src'].apply(lambda x: int(x.split('.')[-1]))
df['dst'] = df['dst'].apply(lambda x: int(x.split('.')[-1]))

# Update values in the specified column based on multiple conditions using apply() and lambda function
df["Protocol"] = df['Protocol'].apply(lambda x: 1 if x == 'UDP' else (2 if x == 'TCP' else 3))

   
# Print the updated DataFrame
print(df)

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_data.csv', index=False)