from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import matplotlib.pyplot as plt
from os import path
from pydub import AudioSegment
import datetime
import threading
import logging
import sys
import pandas as pd 
import numpy as np
import requests
import os

max_len = 100
offset = 1300
_range = 25
thread_num = 4
log_interval = 1
interval = 0

var_list  = [[None]*34]*max_len
mean_list = [[None]*34]*max_len

column_names = []
for i in range(0,34):
    column_names.append('feature ' + str(i))

def function(row, column):
    global interval
    interval += 1
    
    url = df["Episode {}".format(column)][row]
    
    if url is None:
        return
    
    if row == 11861:
        return
    
    mp3 = '{}{}.mp3'.format(row, column)
    wav = '{}{}.wav'.format(row, column)
    
    r = requests.get(url, allow_redirects=True)
    open(mp3, 'wb').write(r.content)
    
    # Export mp3 to wav and remove mp3
    sound = AudioSegment.from_mp3(mp3)
    sound.export(wav, format="wav")    
    os.remove(mp3)

    # Read wav info and remove it
    [Fs, x] = audioBasicIO.read_audio_file(wav)
    if len(x.shape) == 2:
        x = np.mean(x, axis = 1)
    os.remove(wav)
    
    # Extract features
    print("Start {}{} at {}".format(row, column, datetime.datetime.now().time()))
    F = 0
    f_names = 0
    if len(x) > 6*Fs*60:
        x = x[5*Fs*60:6*Fs*60]
        
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
        
    _var = []
    _mean = []
    for f in F:
        _var.append(f.var())
        _mean.append(f.mean())
            
    var_list[row-offset] = _var
    mean_list[row-offset] = _mean
            
    print("End {}{} at {}".format(row, column, datetime.datetime.now().time()))
    
    if interval % 2 == 0:
        pd.DataFrame(var_list,columns=column_names).to_csv(r'./vars{}.csv'.format(offset), index = False, header=True)
        pd.DataFrame(mean_list,columns=column_names).to_csv(r'./means{}.csv'.format(offset), index = False, header=True)
    
df = pd.read_csv('episodes_info.csv')

def thread_function(name):
    print(offset)
    _from = name * _range
    _to = (name+1)* _range
        
    if(_to > max_len):
        _to = max_len
    
    print("from: " + str(_from) + " to: " + str(_to))
        
    for i in range(_from, _to):
        try:
            if(i%10 == 0):
                print(str(_from) + " --> " + str(i))
            function(offset+i, 1)
        except:
            print('\033[91m' + str(offset+i) + ': Page: ' + df['url'][offset+i] + ' was deleted.\033[0m')
            do_something_with_exception()

def do_something_with_exception():
    exc_type, exc_value = sys.exc_info()[:2]
    print ('Handling %s exception with message in %s' % \
        (exc_type.__name__, threading.current_thread().name))
    

print(datetime.datetime.now().time())
            
threads = list()
for index in range(thread_num):
    x = threading.Thread(target=thread_function, args=(index,))
    threads.append(x)
    x.start()
    
for index, thread in enumerate(threads):
    logging.info("Main    : before joining thread %d.", index)
    thread.join()
    logging.info("Main    : thread %d done", index)
    
print(datetime.datetime.now().time())
    
# var = pd.DataFrame(var_list,columns=column_names)
# mean = pd.DataFrame(mean_list,columns=column_names)

# var.to_csv(r'./vars.csv', index = False, header=True)
# mean.to_csv(r'./means.csv', index = False, header=True)

pd.DataFrame(var_list,columns=column_names).to_csv(r'./vars{}.csv'.format(offset), index = False, header=True)
pd.DataFrame(mean_list,columns=column_names).to_csv(r'./means{}.csv'.format(offset), index = False, header=True)