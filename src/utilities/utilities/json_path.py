import json 
path = '/jmain02/home/J2AD007/txk47/sxs27-txk47/LHGNN/datafiles/audioset_bal_tr.json'
out_path = '/jmain02/home/J2AD007/txk47/sxs27-txk47/LHGNN/datafiles/audioset_bal_tr_updated.json'
replace_path = '/jmain02/flash/share/datasets/audioset'
with open(path, 'r') as f:
    data = json.load(f)

for item in data['data']:
    item['wav'] = item['wav'].replace('/data/EECS-MachineListeningLab/datasets/AudioSet/audios', replace_path)

with open(out_path, 'w') as f:
    json.dump(data, f)
