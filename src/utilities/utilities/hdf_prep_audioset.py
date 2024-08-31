import librosa
import soundfile as sf
import h5py
import numpy as np
import json
import os
import csv
import argparse
import sys
from multiprocessing import Pool


json_path = '/jmain02/home/J2AD007/txk47/sxs27-txk47/LHGNN/datafiles/audioset_eval_jade.json'
hdf_path = '/jmain02/home/J2AD007/txk47/sxs27-txk47/datasets/audioset/eval_segments/eval_segments_16k.h5'



def process_audio_item(item, target_sr=16000):
    try:
        file_path = item['wav']
        labels = item['labels']
        print(f"Processing {file_path}")
        print(f"Processing {labels}")
        sys.stdout.flush()
        # Load and resample the audio
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        return audio.astype(np.float32), labels
    except Exception as e:
        print(f"Failed to process {item['wav']}: {e}")
        return None, None

def create_hdf5_dataset(data_json, hdf5_filename, target_sr=16000,num_workers=4):
    # Load the JSON data
    with open(data_json, 'r') as fp:
        data = json.load(fp)['data']
    
    # Open HDF5 file for writing
    with h5py.File(hdf5_filename, 'w') as hdf5_file:
        with Pool(num_workers) as pool:
            results = pool.starmap(process_audio_item, [(item, target_sr) for item in data])
        processed_count = 0
        for i,(audio,labels) in enumerate(results):
            if audio is None or labels is None:
                continue
            
            audio_dataset_name = f'audio_{i}'
            hdf5_file.create_dataset(
                audio_dataset_name,
                data=audio.astype(np.float32),
                compression="gzip",  # Compress the data
                  # Enable chunking for better I/O performance
            )
            # Store the labels as an attribute of the dataset
            hdf5_file[audio_dataset_name].attrs['labels'] = labels
            processed_count += 1

        
    
    print(f"Dataset stored in {hdf5_filename}")
    print(f"Total files processed: {processed_count}")
    sys.stdout.flush()

# Example usage
create_hdf5_dataset(json_path, hdf_path)
