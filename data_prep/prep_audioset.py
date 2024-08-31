import numpy as np
import json
import os

import numpy as np
import json
import os

json_path = '/jmain02/home/J2AD007/txk47/sxs27-txk47/LHGNN/datafiles/audioset_bal_tr.json'
bask_path = '/jmain02/flash/share/datasets/audioset/balanced_train_segments/'
new_json_path = '/jmain02/home/J2AD007/txk47/sxs27-txk47/LHGNN/datafiles/audioset_bal_tr_jade.json'
with open(json_path, 'r') as file:
  data_json = json.load(file)
  counter = 0
  for item in data_json['data']:
     
    item['wav'] = item['wav'].replace('/data/EECS-MachineListeningLab/datasets/AudioSet/audios/balanced_train_segments/', bask_path)

with open(new_json_path, 'w') as file:
    json.dump(data_json, file, indent=1)
  



    

    
    
    
    

        
       
       
    

  