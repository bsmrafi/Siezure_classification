import os
import mne
import numpy as np
#from Seizure_detection_GNN import GGNPipeline
import re
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pickle

import re

def parse_seizures(seizures_file):
    """
    Parse seizure annotations from CHB-MIT text listing.

    Each block looks like:
        File Name: chb01_26.edf
        File Start Time: 12:34:22
        File End Time: 13:13:07
        Number of Seizures in File: 1
        Seizure Start Time: 1862 seconds
        Seizure End Time: 1963 seconds
    Returns a list of dicts, one per seizure.
    """
    with open(seizures_file, "r") as f:
        content = f.read()

    # Split into file-wise blocks
    blocks = re.split(r"\n\s*\n", content.strip())   # blank line separates files
    seizures = []

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        info = { "file_name": None,
                 "file_start_time": None,
                 "file_end_time": None,
                 "seizure_start_sec": None,
                 "seizure_end_sec": None }

        for line in lines:
            if line.lower().startswith("file name"):
                info["file_name"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("file start time"):
                info["file_start_time"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("file end time"):
                info["file_end_time"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("seizure start time"):
                # keep only the numeric seconds before any text
                info["seizure_start_sec"] = float(
                    re.search(r"([\d.]+)", line.split(":", 1)[1]).group(1)
                )
            elif line.lower().startswith("seizure end time"):
                info["seizure_end_sec"] = float(
                    re.search(r"([\d.]+)", line.split(":", 1)[1]).group(1)
                )

        # Add one entry for every seizure reported
        if info["seizure_start_sec"] is not None and info["seizure_end_sec"] is not None:
            seizures.append(info)

    return seizures


def parse_time_to_seconds(time_str, registration_start=None):
    """Convert various time formats to seconds from start of recording"""
    if not time_str:
        return 0
    
    time_str = time_str.strip()
    time_str = re.sub(r'\s+', '', time_str)
    
    # Clean up any extra text
    time_str = time_str.split('(')[0].strip()
    
    for sep in [':', '.', ' ']:
        if sep in time_str:
            parts = time_str.split(sep)
            break
    else:
        if len(time_str) == 6:
            parts = [time_str[0:2], time_str[2:4], time_str[4:6]]
        else:
            print(f"Warning: Cannot parse time format: {time_str}")
            return 0
    
    try:
        if len(parts) == 3:
            h, m, s = map(float, parts)
        elif len(parts) == 2:
            if float(parts[0]) > 23:
                h = 0
                m, s = map(float, parts)
            else:
                h, m = map(float, parts)
                s = 0
        else:
            print(f"Warning: Unexpected time format: {time_str}")
            return 0
        
        total_seconds = h * 3600 + m * 60 + s
        
        if registration_start:
            reg_seconds = parse_time_to_seconds(registration_start)
            relative_seconds = total_seconds - reg_seconds
            if relative_seconds < 0:
                relative_seconds += 24 * 3600
            return relative_seconds
        
        return total_seconds
        
    except Exception as e:
        print(f"Error parsing time {time_str}: {e}")
        return 0

def load_chbmit_dataset(base_dir, sampling_rate=512):
    """Load entire Siena Scalp EEG Database"""
    
    seizure_data = []
    non_seizure_data = []
    
    patient_folders = [f for f in os.listdir(base_dir) if f.startswith('chb')]
    patient_folders.sort()
    
    print(f"Found {len(patient_folders)} patient folders")
    
    for patient_folder in patient_folders:
        patient_path = os.path.join(base_dir, patient_folder)
        seizures_file = os.path.join(patient_path, f"{patient_folder}-summary.txt")
       # print(seizures_file);exit(0)
        if not os.path.exists(seizures_file):
            print(f"Warning: No seizure list found for {patient_folder}-summary")
            continue
            
        print(f"\nProcessing {patient_folder}...")
        #print(seizures_file);exit(0)
        seizures = parse_seizures(seizures_file)
        #print(seizures);exit(0)
        if not seizures:
            print(f"  No seizures found for {patient_folder}")
            continue
        
        for seizure_info in seizures:
            file_name = seizure_info['file_name']
            
            # Fix typos in file names
            #file_name = file_name.replace('chbO6', 'chb06')
            
            # Find the EDF file
            edf_file = os.path.join(patient_path, file_name)
            if not os.path.exists(edf_file):
                # Try without .edf extension in the middle
                file_name_fixed = file_name.replace('.edf', '')
                edf_file = os.path.join(patient_path, f"{file_name_fixed}.edf")
                if not os.path.exists(edf_file):
                    print(f"  Warning: EDF file not found: {file_name}")
                    continue
            
            try:
                raw_data = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
                actual_sfreq = raw_data.info['sfreq']
                #raw_data.resample(256)
                # Get all EEG channels (exclude EKG, ECG, EOG)
                eeg_channels = []
                for ch in raw_data.info['ch_names']:
                    ch_upper = ch.upper()
                    # Include only EEG channels
                    if 'EEG' in ch_upper or any(label in ch_upper for label in ['FP', 'F', 'C', 'P', 'O', 'T']):
                        if not any(exclude in ch_upper for exclude in ['EKG', 'ECG', 'EOG', 'EMG']):
                            eeg_channels.append(ch)
                
                if len(eeg_channels) < 10:
                    print(f"  Warning: Only {len(eeg_channels)} EEG channels found")
                    continue
                
                # Use the newer pick() method instead of pick_channels()
                raw_data = raw_data.pick(eeg_channels)
                #print(f"  Using {len(eeg_channels)} EEG channels")
                
                # Convert times to seconds
                start_seconds = seizure_info['seizure_start_sec']
                end_seconds   = seizure_info['seizure_end_sec']
                
                # Convert to samples
                start_sample = int(start_seconds * actual_sfreq)
                end_sample = int(end_seconds * actual_sfreq)
                #print(edf_file,start_sample,end_sample,start_seconds,end_seconds,raw_data.get_data().shape)
                # Extract seizure segment
                if 0 <= start_sample < raw_data.n_times and end_sample <= raw_data.n_times and end_sample > start_sample:
                    seizure_segment = raw_data.get_data(start=start_sample, stop=end_sample)
                    seizure_data.append(seizure_segment)
                    duration = (end_sample - start_sample) / actual_sfreq
                    print(f"  ✓ Extracted seizure: {duration:.1f}s ({seizure_segment.shape[0]} channels)")
                else:
                    print(f"  ✗ Sample indices out of range: {start_sample} - {end_sample} (max: {raw_data.n_times})")
                
                # Extract baseline (30s from beginning if safe)
                #print(actual_sfreq)
                baseline_duration = int(30 * actual_sfreq)
                if start_sample > baseline_duration:
                    baseline_segment = raw_data.get_data(start=0, stop=baseline_duration)
                    #print(baseline_segment.shape)
                    non_seizure_data.append(baseline_segment)
                    #print(non_seizure_data[-1].shape)
                    print(f"  ✓ Extracted baseline segment for ({baseline_duration} =30 secs)")
                    
            except Exception as e:
                print(f"  Error processing {file_name}: {e}")
                continue
    #print("ns",non_seizure_data[1][1].shape)
    all_data = seizure_data + non_seizure_data #first 23 are siezures, next 23 are non seizures
    all_labels = [1] * len(seizure_data) + [0] * len(non_seizure_data)
    #for w in range(0,len(all_data)):
    #    print(all_data[w].shape)
    print(f"\n{'='*50}")
    print(f"Dataset Summary:")
    print(f"  Total segments: {len(all_data)}")
    print(f"  Seizures: {len(seizure_data)}")
    print(f"  Non-seizures: {len(non_seizure_data)}")
    
    return all_data, all_labels

if __name__ == '__main__':
# Specify Siena database directory
    base_dir = "/home/rafi/physionet.org/files/chbmit/1.0.0"  # Add path to data

    data, labels = load_chbmit_dataset(base_dir)

    obj_data = np.empty(len(data), dtype=object)
    for i, seg in enumerate(data):
        obj_data[i] = seg   # each entry is its own ndarray

    np.savez_compressed("chbmit_eeg_segments.npz",data=obj_data,labels=np.array(labels, dtype=np.int8))
        
    loaded = np.load("chbmit_eeg_segments.npz", allow_pickle=True)
    data_segments = loaded["data"]      # returns object array of segments
    labels = loaded["labels"]
    
 #   print(data_segments.shape)
  #  print(labels)
