import pickle
f = open('./plot/SOCIALLSTM/LSTM/validation/biwi/biwi_hotel_4.pkl', 'rb')
info = pickle.load(f)
print(info)