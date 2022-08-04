from scipy.io import loadmat
import pandas as pd

matlabfile = loadmat("data_EEG_AI.mat")

print(matlabfile.keys())
labels = ['channel_labels', 'data', 'label', 'time_points']

data_dict = dict()
channel_label_list = [] 

# 1. store the name of the label to the channel_label_list
for i in matlabfile['channel_labels']:
    channel_label_list.append(str(list(i[0])[0]))
print(channel_label_list)

# 2. construct the data_dict

no_of_channel = len(channel_label_list) # just 24 channels 
no_of_sample_per_alphabet = 300
no_of_timepoints = 801 

for k, channel_label in enumerate(channel_label_list):
# for i, element in enumerate(seq):
    temp_channel_list = []
    for i in range(0, 26*no_of_sample_per_alphabet):
        temp_timepoint_list = []
        for j in range(0, no_of_timepoints):
            temp_timepoint_list.append(matlabfile["data"][k][j][i])

        temp_channel_list.append(temp_timepoint_list)

    data_dict[channel_label] = temp_channel_list
    print("Finish 1 channel")

# 3. convert data_dict to a pandas dataframe 

pandas_df = pd.DataFrame.from_dict(data_dict)
print(pandas_df.info())
row2 = pandas_df.iloc[3][0]
print(len(row2))

# 4. Export the pandas dataframe to be .txt file
pandas_df.to_csv('data.txt', sep=',')