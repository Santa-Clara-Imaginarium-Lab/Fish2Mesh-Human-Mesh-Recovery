import pickle

def load_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Example usage
file_path = '/media/imaginarium/12T/Dataset/EgoGTA/test/jian1/annotation.pkl'
file_path = '/home/imaginarium/Downloads/archive/data.pkl'
data = load_pkl_file(file_path)

# Print the loaded data
print(data)
