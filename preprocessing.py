import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load raw data
raw_data_df = pd.read_csv("data/responses.csv")

# Remove spaces from and convert column headings to lowercase
raw_data_df.columns = [x.lower().replace(" ", "") for x in raw_data_df.columns]

# Drop row with missing responses in gender or age
raw_data_df = raw_data_df.dropna(subset=['gender', 'age'])

# Convert strings columns to category series with codes
for col in ['house-blockofflats', 'village-town', 'onlychild', 'education', 'left-righthanded', 'gender',
            'internetusage', 'lying', 'punctuality', 'alcohol', 'smoking']:
  raw_data_df[col] = raw_data_df[col].astype('category').cat.codes

# Select only columns which are intuitively the best predictors of gender/age to avoid over-fitting model
raw_data_df = raw_data_df[['gender', 'age', 'pop', 'opera', 'horror', 'romantic', 'sci-fi', 'documentary', 'history',
                           'politics', 'cars', 'artexhibitions', 'gardening', 'celebrities', 'theatre', 'darkness',
                           'spiders', 'ageing', 'smoking', 'alcohol','healthyeating', 'thinkingahead',
                           'criminaldamage', 'elections', 'giving', 'lifestruggles', 'internetusage', 'finances',
                           'spendingonlooks']]

# Fill in NaNs with column means
raw_data_df = raw_data_df.fillna(raw_data_df.mean())

# Split raw data into training and testing sets
train_data_df, test_data_df = train_test_split(raw_data_df, test_size=0.2)

# Scale training and testing data between 0 and 1
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_training = scaler.fit_transform(train_data_df)
scaled_testing = scaler.transform(test_data_df)

# Export data needed to un-scale results
m = pd.Series(data=scaler.scale_, index=test_data_df.columns.values)
c = pd.Series(data=scaler.min_, index=test_data_df.columns.values)
scaling = pd.concat([m, c], axis=1).T
scaling.to_csv("model/scaling.csv")

# Convert scaled arrays back to dataframes
scaled_training_df = pd.DataFrame(scaled_training, columns=train_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

# Export to CSV
scaled_training_df.to_csv("model/training_scaled.csv", index=False)
scaled_testing_df.to_csv("model/test_scaled.csv", index=False)