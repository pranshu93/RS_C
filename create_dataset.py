import pickle 
import soundfile as sf
import numpy as np
import os
import librosa 

data_dir = os.path.abspath("../RS_RAF")

cl_dir_names = ['wheezes','sneezing','cough','snoring','wheezes_crackles','breathing','crackles']
# /RS_RAF/cough/Coswara/

conv_fac = 441

data = {}
data['wheezes'] = None
data['sneezing'] = {}
data['sneezing']['ESC50'] = None
data['cough'] = {}
data['cough']['Coswara'] = None
data['cough']['ESC50'] = None
data['cough']['Soundsnap'] = None
data['snoring'] = {}
data['wheezes_crackles'] = None
data['breathing'] = {}
data['breathing']['ESC50'] = None
data['crackles'] = None

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

def uniformize(obj):
    global conv_fac
    new_obj = []
    arr_len = lengths(obj)
    max_len = np.max(arr_len)
    uni_len = int(np.ceil(max_len/conv_fac)*conv_fac)
    for row in obj:
        #new_obj.append(np.append(row,np.zeros(uni_len-len(row)),0).tolist())
        new_obj.append(np.append(np.zeros(uni_len-len(row)),row,0).tolist())
    return np.array(new_obj)

def aggregate(obj):
    global conv_fac
    return np.average(obj.reshape((obj.shape[0],-1,conv_fac)),axis=2)

def read_data(path):
    arr = []
    cnt1 = 0
    cnt2 = 0
    for filename in os.listdir(path):
        ext = os.path.splitext(filename)[-1].lower()
        if(ext == ".ogg" or ext == ".wav"):
            values, samplerate = librosa.load(os.path.join(path,filename),sr=44100)
            #values, samplerate = sf.read(os.path.join(path,filename))
            if(samplerate == 44100):
                cnt1 += 1
            else:
                cnt2 += 1
            
            #print(samplerate)
            #assert(samplerate==44100)
            arr.append(values.tolist())
    
    arr = np.array(arr,dtype=object)
    print(cnt1,cnt2)
    return arr;

def read_data_Coswara():
    arr = []
    cnt1 = 0
    cnt2 = 0
    folder_names = ['20200413','20200416','20200418','20200424','20200502','20200505','20200604','20200720','20200814','20200824','20200415','20200417','20200419','20200430','20200504','20200525','20200707','20200803','20200820','20200901']
    for folder in folder_names:
        path = os.path.join(data_dir,'cough','Coswara',folder,folder)
        for filename in os.listdir(path):
            if os.path.isdir(os.path.join(path,filename)):
                #values, samplerate = sf.read(os.path.join(path,filename,'cough-heavy.wav'))
                try:
                    values, samplerate = librosa.load(os.path.join(path,filename,'cough-heavy.wav'),sr=44100)
                    if(samplerate == 44100):
                        cnt1 += 1
                    else:
                        cnt2 += 1
                
    #assert(samplerate==44100)
                    arr.append(values.tolist())
                except:
                    continue
    arr = np.array(arr,dtype=object)
    print(cnt1,cnt2)
    return arr;


# breathing - ESC50
# cough - Coswara, ESC50, Soundsnap
# snoring - ESC50
'''
data['wheezes'] = read_data(os.path.abspath("../RS_RAF/wheezes"))
print(data['wheezes'].shape)
data['sneezing']['ESC50'] = read_data(os.path.abspath("../RS_RAF/sneezing/ESC-50"))
print(data['sneezing']['ESC50'].shape)
data['cough']['Coswara'] = read_data_Coswara()
print(data['cough']['Coswara'].shape)
data['cough']['ESC50'] = read_data(os.path.abspath("../RS_RAF/cough/ESC-50"))
print(data['cough']['ESC50'].shape)
data['cough']['Soundsnap'] = read_data(os.path.abspath("../RS_RAF/cough/soundsnap"))
print(data['cough']['Soundsnap'].shape)
data['snoring']['ESC50'] = read_data(os.path.abspath("../RS_RAF/snoring/ESC-50"))
print(data['snoring']['ESC50'].shape)
data['wheezes_crackles'] = read_data(os.path.abspath("../RS_RAF/wheezes_crackles"))
print(data['wheezes_crackles'].shape)
data['breathing']['ESC50'] = read_data(os.path.abspath("../RS_RAF/breathing/ESC-50"))
print(data['breathing']['ESC50'].shape)
data['crackles'] = read_data(os.path.abspath("../RS_RAF/crackles/"))
print(data['crackles'].shape)
'''
data['wheezes'] = aggregate(uniformize(read_data(os.path.abspath("../RS_RAF/wheezes"))))
print(data['wheezes'].shape)
#'''
data['sneezing']['ESC50'] = aggregate(uniformize(read_data(os.path.abspath("../RS_RAF/sneezing/ESC-50"))))
print(data['sneezing']['ESC50'].shape)
data['cough']['Coswara'] = aggregate(uniformize(read_data_Coswara()))
print(data['cough']['Coswara'].shape)
data['cough']['ESC50'] = aggregate(uniformize(read_data(os.path.abspath("../RS_RAF/cough/ESC-50"))))
print(data['cough']['ESC50'].shape)
data['cough']['Soundsnap'] = aggregate(uniformize(read_data(os.path.abspath("../RS_RAF/cough/soundsnap"))))
print(data['cough']['Soundsnap'].shape)
data['snoring']['ESC50'] = aggregate(uniformize(read_data(os.path.abspath("../RS_RAF/snoring/ESC-50"))))
print(data['snoring']['ESC50'].shape)
data['wheezes_crackles'] = aggregate(uniformize(read_data(os.path.abspath("../RS_RAF/wheezes_crackles"))))
print(data['wheezes_crackles'].shape)
data['breathing']['ESC50'] = aggregate(uniformize(read_data(os.path.abspath("../RS_RAF/breathing/ESC-50"))))
print(data['breathing']['ESC50'].shape)
data['crackles'] = aggregate(uniformize(read_data(os.path.abspath("../RS_RAF/crackles/"))))
print(data['crackles'].shape)
#'''
#print(data['cough']['Coswara'].shape)
#print(data['sneezing']['ESC50'].shape)

  
#  dictionary = {'geek': 1, 'supergeek': True, 4: 'geeky'} 


try: 
    geeky_file = open('data.pkl', 'wb') 
    pickle.dump(data, geeky_file) 
    geeky_file.close() 
                  
except: 
    print("Something went wrong")

