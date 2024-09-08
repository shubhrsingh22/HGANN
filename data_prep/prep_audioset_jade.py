import numpy as np 
import json 
import os 
import re

root_path = "/jmain02/home/J2AD007/txk47/sxs27-txk47/LHGNN/jade_meta"
meta_path = os.path.join(root_path, "meta")
audio_path = '/jmain02/flash/share/datasets/audioset/'


#path = "/data/EECS-MachineListeningLab/datasets/AudioSet/meta/custom.all_train_segments.csv"
bal_tr_meta = os.path.join(meta_path,"custom.balanced_train_segments.csv")
all_tr_meta = os.path.join(meta_path,"custom.all_train_segments.csv")
eval_meta = os.path.join(meta_path,"custom.eval_segments.csv")

bal_tr_meta = np.loadtxt(bal_tr_meta, skiprows=1,dtype=str)
all_tr_meta = np.loadtxt(all_tr_meta, skiprows=1,dtype=str)
eval_meta = np.loadtxt(eval_meta, skiprows=1,dtype=str)

out_path = '/jmain02/home/J2AD007/txk47/sxs27-txk47/LHGNN/datafiles'

tr_cnt, eval_cnt = 0, 0

bal_tr_data = []
eval_data = []
all_tr_data = []

bal_cnt = 0
tr_cnt = 0
eval_cnt = 0

def convert_part_code(path):
    # Use regex to find 'partXX' where XX is any two digits and convert it
    return re.sub(r'part0+(\d)', r'part\1', path)
# for i in range(len(bal_tr_meta)):
    
#     try:
#         #fileid = bal_tr_meta[i].split('|')[0]
#         fileid = bal_tr_meta[i].split('.')[0]
#         labels = bal_tr_meta[i].split('|')[1]
     
        
#     except:
#         fileid = bal_tr_meta[i].split('.')[0]
#         labels = bal_tr_meta[i].split('|')[1]
      
    
    
   
#     labels = labels.split('",')[0]
    
#     label_list = labels.split(',')
    
#     new_label_list = []
#     for label in label_list:
#         new_label_list.append(label)
    
#     new_label_list = ','.join(new_label_list)
    
#     wav_path = root_path + 'audios/'+ fileid + '.wav'
#     if os.path.exists(wav_path):
#         cur_dict = {"wav": root_path + 'audios/'+ fileid + '.wav',
#                     "labels": new_label_list}

#         bal_tr_data.append(cur_dict)
#         bal_cnt += 1
#     print(f'total bal count: {bal_cnt}')
    

for i in range(len(all_tr_meta)):
    try:
        #fileid = bal_tr_meta[i].split('|')[0]
        fileid = all_tr_meta[i].split('.')[0]
        labels = all_tr_meta[i].split('|')[1]

        if 'unbalanced_train_segments/' in fileid:
            fileid = fileid.split('unbalanced_train_segments/')[1]
            fileid = convert_part_code(fileid)
            print(fileid)

    
    except:
        fileid = all_tr_meta[i].split('.')[0]
        if 'unbalanced_train_segments/' in fileid:
            fileid = fileid.split('unbalanced_train_segments/')[1]
            print(fileid)

        labels = all_tr_meta[i].split('|')[1]




    labels = labels.split('",')[0]

    label_list = labels.split(',')

    new_label_list = []
    for label in label_list:
        new_label_list.append(label)

    new_label_list = ','.join(new_label_list)
    wav_path = wav_path = audio_path + fileid + '.wav'
    if os.path.exists(wav_path):
        cur_dict = {"wav": wav_path,
                    "labels": new_label_list}
        all_tr_data.append(cur_dict)
        tr_cnt += 1
    print(f'total training count: {tr_cnt}')
    

# for i in range(len(eval_meta)):
#     try:
#         #fileid = bal_tr_meta[i].split('|')[0]
#         fileid = eval_meta[i].split('.')[0]
#         labels = eval_meta[i].split('|')[1]
#     except:
#         fileid = eval_meta[i].split('.')[0]
#         labels = eval_meta[i].split('|')[1]
    
#     labels = labels.split('",')[0]

#     label_list = labels.split(',')

#     new_label_list = []
#     for label in label_list:
#         new_label_list.append(label)

#     new_label_list = ','.join(new_label_list)
#     wav_path = audio_path + fileid + '.wav'
#     if os.path.exists(wav_path):

#         cur_dict = {"wav": wav_path,
#                 "labels": new_label_list}

#         eval_data.append(cur_dict)
#         eval_cnt += 1
#     print(f'total eval count: {eval_cnt}')
    
datafile_path = out_path
# #tr_json = os.path.join(datafile_path, 'audioset_bal_tr.json')
all_tr_json = os.path.join(datafile_path, 'audioset_all_tr.json')
# eval_json = os.path.join(datafile_path, 'audioset_eval_jade.json')



#if not os.path.exists(datafile_path):
#    os.makedirs(datafile_path)

#with open(tr_json, 'w') as file:
#    json.dump({"data": bal_tr_data}, file, indent=1)

with open(all_tr_json, 'w') as file:
    json.dump({"data": all_tr_data}, file, indent=1)

#with open(eval_json, 'w') as file:
#    json.dump({"data": eval_data}, file, indent=1)

#print('Processed {:d} samples for the Balanced training set '.format(bal_cnt))
print('Processed {:d} samples for the Entire training set'.format(tr_cnt))
#print('Processed {:d} samples for the Evaluation set.'.format(eval_cnt))


    



