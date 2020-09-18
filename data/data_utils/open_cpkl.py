import pickle
f = open('../train/trajectories_train.cpkl', 'rb+')
info = pickle.load(f)
print(info)