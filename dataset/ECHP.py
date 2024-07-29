import scipy.io

# Define the path to the .mat file
mat_path = '/media/imaginarium/12T/Dataset/ECHP/Info/env1/'
mat_file = '00201.mat'
full_path = mat_path + mat_file

# Load the .mat file
data = scipy.io.loadmat(full_path)

# Print the keys in the loaded data to see what variables are available
print(data.keys())