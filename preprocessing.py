import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load raw data
raw_data_df = pd.read_csv("data/responses.csv")

# Remove spaces from and convert column headings to lowercase
raw_data_df.columns = [x.lower().replace(" ", "") for x in raw_data_df.columns ]

# Convert string to categories codes
for col in ['house-blockofflats', 'village-town', 'onlychild', 'education', 'left-righthanded', 'gender',
            'internetusage', 'lying', 'punctuality', 'alcohol', 'smoking']:
  raw_data_df[col] = raw_data_df[col].astype('category').cat.codes

# Fill in NaNs with column means
raw_data_df = raw_data_df.fillna(raw_data_df.mean())

# Split dataframe into training and testing sets
train_data_df, test_data_df = train_test_split(raw_data_df, test_size=0.2)

# Scale training and testing data between 0 and 1
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_training = scaler.fit_transform(train_data_df)
scaled_testing = scaler.transform(test_data_df)

# Export age scaling values as CSV
with open('model/agescale.csv','w') as f:
  f.write(str(scaler.scale_[140]))
  f.write(',')
  f.write(str(scaler.min_[140]))

# Convert scaled arrays back to dataframes
scaled_training_df = pd.DataFrame(scaled_training, columns=train_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

# Un-scale gender column so values are 0 (female) and 1 (male) rather than 0.5 and 1
scaled_training_df['gender'] = (scaled_training_df['gender'] - 0.5) * 2
scaled_testing_df['gender'] = (scaled_testing_df['gender'] - 0.5) * 2

# Export to CSV
scaled_training_df.to_csv("model/training_scaled.csv", index=False)
scaled_testing_df.to_csv("model/test_scaled.csv", index=False)