import os
import mne
import numpy as np
#from Seizure_detection_GNN import GGNPipeline
import re
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def parse_seizures(seizures_file):
    """Parse seizure data from various Siena format files"""
    with open(seizures_file, 'r') as f:
        content = f.read()
    
    seizures = []
    
    # Split by "Seizure n" to find each seizure block
    seizure_blocks = re.split(r'Seizure n\s*\d+', content)[1:]
    seizure_numbers = re.findall(r'Seizure n\s*(\d+)', content)
    
    for i, block in enumerate(seizure_blocks):
        seizure_info = {
            'seizure_num': int(seizure_numbers[i]) if i < len(seizure_numbers) else i+1,
            'file_name': None,
            'seizure_start_time': None,
            'seizure_end_time': None,
            'registration_start_time': None
        }
        
        lines = block.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if 'File name:' in line or ('edf' in line.lower() and 'File' in line):
                match = re.search(r'(PN[O0-9]+-[\d.]+\.edf)', line)  # Handle PNO6 typo
                if match:
                    seizure_info['file_name'] = match.group(1)
                else:
                    match = re.search(r'([^\s]+\.edf)', line)
                    if match:
                        seizure_info['file_name'] = match.group(1)
            
            elif 'Registration start time' in line:
                time_part = line.split(':', 1)[1].strip() if ':' in line else line.split('time', 1)[1].strip()
                seizure_info['registration_start_time'] = time_part
            
            elif ('Start time' in line or 'Seizure start time' in line) and 'Registration' not in line:
                time_part = line.split(':', 1)[1].strip() if ':' in line else line.split('time', 1)[1].strip()
                # Clean up the time string
                time_part = time_part.split('(')[0].strip()  # Remove anything after parenthesis
                seizure_info['seizure_start_time'] = time_part
            
            elif ('End time' in line or 'Seizure end time' in line) and 'Registration' not in line:
                time_part = line.split(':', 1)[1].strip() if ':' in line else line.split('time', 1)[1].strip()
                time_part = time_part.split()[0].strip()  # Take only first part if multiple times
                seizure_info['seizure_end_time'] = time_part
        
        if not seizure_info['registration_start_time']:
            header_match = re.search(r'Registration start time[:\s]+([0-9.:]+)', content[:content.find('Seizure n')])
            if header_match:
                seizure_info['registration_start_time'] = header_match.group(1)
        
        if seizure_info['file_name'] and seizure_info['seizure_start_time'] and seizure_info['seizure_end_time']:
            seizures.append(seizure_info)
            print(f"  Found seizure {seizure_info['seizure_num']}: {seizure_info['file_name']}")
            print(f"    Start: {seizure_info['seizure_start_time']}, End: {seizure_info['seizure_end_time']}")
    
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

def load_siena_dataset(base_dir, sampling_rate=512):
    """Load entire Siena Scalp EEG Database"""
    
    seizure_data = []
    non_seizure_data = []
    
    patient_folders = [f for f in os.listdir(base_dir) if f.startswith('PN')]
    patient_folders.sort()
    
    print(f"Found {len(patient_folders)} patient folders")
    
    for patient_folder in patient_folders:
        patient_path = os.path.join(base_dir, patient_folder)
        seizures_file = os.path.join(patient_path, f"Seizures-list-{patient_folder}.txt")
        
        if not os.path.exists(seizures_file):
            print(f"Warning: No seizure list found for {patient_folder}")
            continue
            
        print(f"\nProcessing {patient_folder}...")
        seizures = parse_seizures(seizures_file)
        #print(seizures);exit(0);
        if not seizures:
            print(f"  No seizures found for {patient_folder}")
            continue
        
        for seizure_info in seizures:
            file_name = seizure_info['file_name']
            
            # Fix typos in file names
            file_name = file_name.replace('PNO6', 'PN06')
            
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
                start_seconds = parse_time_to_seconds(
                    seizure_info['seizure_start_time'], 
                    seizure_info['registration_start_time']
                )
                end_seconds = parse_time_to_seconds(
                    seizure_info['seizure_end_time'],
                    seizure_info['registration_start_time']
                )
                
                # Convert to samples
                start_sample = int(start_seconds * actual_sfreq)
                end_sample = int(end_seconds * actual_sfreq)
                
                # Extract seizure segment
                if 0 <= start_sample < raw_data.n_times and end_sample <= raw_data.n_times and end_sample > start_sample:
                    seizure_segment = raw_data.get_data(start=start_sample, stop=end_sample)
                    seizure_data.append(seizure_segment)
                    duration = (end_sample - start_sample) / actual_sfreq
                    print(f"  ✓ Extracted seizure: {duration:.1f}s ({seizure_segment.shape[0]} channels)")
                else:
                    print(f"  ✗ Sample indices out of range: {start_sample} - {end_sample} (max: {raw_data.n_times})")
                
                # Extract baseline (30s from beginning if safe)
                baseline_duration = int(30 * actual_sfreq)
                if start_sample > baseline_duration:
                    baseline_segment = raw_data.get_data(start=0, stop=baseline_duration)
                    non_seizure_data.append(baseline_segment)
                    #print(f"  ✓ Extracted baseline segment")
                    
            except Exception as e:
                print(f"  Error processing {file_name}: {e}")
                continue
    
    all_data = seizure_data + non_seizure_data
    all_labels = [1] * len(seizure_data) + [0] * len(non_seizure_data)
    
    print(f"\n{'='*50}")
    print(f"Dataset Summary:")
    print(f"  Total segments: {len(all_data)}")
    print(f"  Seizures: {len(seizure_data)}")
    print(f"  Non-seizures: {len(non_seizure_data)}")
    
    return all_data, all_labels

if __name__ == '__main__':
# Specify Siena database directory
    base_dir = "/home/rafi/physionet.org/files/siena-scalp-eeg/1.0.0"  # Add path to data

    data_segments, labels = load_siena_dataset(base_dir)
    #print(data_segments.shape, labels.shape);exit(0)
    np.savez_compressed("siena_eeg_segments.npz",data=np.array(data_segments, dtype=object),labels=np.array(labels, dtype=np.int8))
    loaded = np.load("siena_eeg_segments.npz", allow_pickle=True)
    data_segments = loaded["data"]      # returns object array of segments
    labels = loaded["labels"]
    
    print(data_segments.shape)
    print(labels)

    
