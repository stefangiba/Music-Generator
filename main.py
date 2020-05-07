#!/usr/bin/python
#importuri utile printe care os care ne permite sa apelam alte aplicatii din calculator
#json pentru parsare
#numpy pentru calcule si afisari
#pandas pentru analiza de date
#si keras care este un framework de tenserflow care usureaza lucrul cu modelele de inteligenta artificiala
import os
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding
import tkinter as tk

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#variabile utile
data_directory = "./Data/"
data_file = "Data_Tunes.txt"
charIndex_json = "char_to_index.json"
model_weights_directory = './Data/Model_Weights/'
BATCH_SIZE = 16
SEQ_LENGTH = 64

#declararea modelului pe care il folosim
#de fiecare data cand se apeleaza metoda add se adauga inca un strat la reteaua neuronala

def make_model(unique_chars):
    model = Sequential()
    model.add(Embedding(input_dim = unique_chars, output_dim = 512, batch_input_shape = (1, 1))) 
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, stateful = True)) 
    #remember, that here we haven't given return_sequences = True because here we will give only one character to generate the
    #sequence. In the end, we just have to get one output which is equivalent to getting output at the last time-stamp. So, here
    #in last layer there is no need of giving return sequences = True.
    model.add(Dropout(0.2))
    model.add((Dense(unique_chars)))
    model.add(Activation("softmax"))
    
    return model

def generate_sequence(epoch_num, initial_index, seq_length):
    with open(os.path.join(data_directory, charIndex_json)) as f:
        char_to_index = json.load(f)
    index_to_char = {i:ch for ch, i in char_to_index.items()}
    unique_chars = len(index_to_char)
    
    model = make_model(unique_chars)
    model.load_weights(model_weights_directory + "Weights_{}.h5".format(epoch_num))
     
    sequence_index = [initial_index]
    
    for _ in range(seq_length):
        batch = np.zeros((1, 1))
        batch[0, 0] = sequence_index[-1]
        
        predicted_probs = model.predict_on_batch(batch).ravel()
        sample = np.random.choice(range(unique_chars), size = 1, p = predicted_probs)
        
        sequence_index.append(sample[0])
    
    seq = ''.join(index_to_char[c] for c in sequence_index)
    
    cnt = 0
    for i in seq:
        cnt += 1
        if i == "\n":
            break
    seq1 = seq[cnt:]
    #above code is for ignoring the starting string of a generated sequence. This is because we are passing any arbitrary 
    #character to the model for generating music. Now, the model start generating sequence from that character itself which we 
    #have passed, so first few characters before "\n" contains meaningless word. Model start generating the music rhythm from
    #next line onwards. The correct sequence it start generating from next line onwards which we are considering.
    
    cnt = 0
    for i in seq1:
        cnt += 1
        if i == "\n" and seq1[cnt] == "\n":
            break
    seq2 = seq1[:cnt]
    #Now our data contains three newline characters after every tune. So, the model has leart that too. So, above code is used for
    #ignoring all the characters that model has generated after three new line characters. So, here we are considering only one
    #tune of music at a time and finally we are returning it..
    
    return seq2

fields = ('Nr. of Epochs', 'Initial character', 'Length')

def makeform(root, fields):
    entries = {}
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=22, text=field+": ", anchor='w')
        ent = tk.Entry(row)
        ent.insert(0, "0")
        row.pack(side=tk.TOP, 
                fill=tk.X, 
                padx=5, 
                pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, 
                expand=tk.YES, 
                fill=tk.X)
        entries[field] = ent

    return entries

def generate_sequence_main(entries):
    ep = int(entries["Nr. of Epochs"].get()) # nr.epoci
    ar = int(entries["Initial character"].get()) # caracterul initial dupa care se genereaza urmatoarele note
    ln = int(entries["Length"].get()) # lungimea sirului muzical

    # print("\n" + str(ep) + " " + str(ar) + " " + str(ln) + "\n")
    
    music = generate_sequence(ep, ar, ln)
    print("\nMUSIC SEQUENCE GENERATED: \n")
    print(music)
    header = """X:1
T:Test
M:6/8
L:1/8
R:jig
K:G
"""
    with open("test.abc","w+") as file:
        file.write(header)
    with open("test.abc","a") as file:
        file.write(music)
    # os.system('cmd /k "abc2midi test.abc -o test.mid"')
    # os.system('cmd /k "\"test3.mid\""')

    
root = tk.Tk()
root.title("Music Generation")
#root.geometry("400x300")
ents = makeform(root, fields)
b1 = tk.Button(root, text='Quit',
       command=root.destroy)
b1.pack(side=tk.LEFT, padx=5, pady=5)
b2 = tk.Button(root, text='Generate',
       command=(lambda e=ents: generate_sequence_main(e)))
b2.pack(side=tk.RIGHT, padx=5, pady=5)
root.mainloop()


