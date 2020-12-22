import pickle 
import numpy as np

f = open('data.pkl', 'rb')   # 'r' for reading; can be omitted
data = pickle.load(f)         # load file content as mydict
f.close()                       

def lengths(obj):
    arr = []
    #num_rows = obj.shape[0]
    for row in obj:
        arr.append(len(row))
    return np.array(arr)

def statistics(vals):
    flat_vals = np.hstack(vals.flatten())
    mean = np.mean(flat_vals)
    std = np.std(flat_vals)
    num_vals = flat_vals.shape[0]
    #length_checker = np.vectorize(len) 
    arr_len = lengths(vals)
    print(arr_len)
    #print(num_vals,mean,std)



statistics(data['wheezes'])
#print(data['wheezes'].shape)
