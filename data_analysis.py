import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
import json
import seaborn as sns
from pathlib import Path

"""
This script analyzes the Lao instrument dataset to understand its characteristics.
It will generate visualizations and statistics that will help us improve the model.
"""

class DatasetAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.results = {
            "instrument_stats": {},
            "duration_stats": {},
            "spectral_stats": {},
            "amplitude_stats": {},
            "sessions": {}
        }
        self.sample_rate = 44100
    
    def analyze_dataset(self):
        """Analyze the full dataset structure and audio characteristics"""
        print(f"Analyzing dataset at {self.data_path}")
        
        # Create output directory
        os.makedirs("analysis_results", exist_ok=True)
        
        # Get all instrument folders
        instrument_folders = [d for d in os.listdir(self.data_path) 
                             if os.path.isdir(os.path.join(self.data_path, d))]
        
        print(f"Found {len(instrument_folders)} instrument folders: {instrument_folders}")
        
        # Map folders to standardized instrument names
        instrument_mapping = {}
        for folder in instrument_folders:
            if 'khean' in folder.lower() or 'khaen' in folder.lower():
                instrument_mapping[folder] = 'khean'
            elif 'khong' in folder.lower() or 'kong' in folder.lower():
                instrument_mapping[folder] = 'khong_vong'
            elif 'pin' in folder.lower():
                instrument_mapping[folder] = 'pin'
            elif 'nad' in folder.lower() or 'ranad' in folder.lower():
                instrument_mapping[folder] = 'ranad'
            elif 'saw' in folder.lower() or 'so' in folder.lower():
                instrument_mapping[folder] = 'saw'
            elif 'sing' in folder.lower():
                instrument_mapping[folder] = 'sing'
            elif 'background' in folder.lower() or 'noise' in folder.lower():
                instrument_mapping[folder] = 'background'
            else:
                instrument_mapping[folder] = folder.lower()
        
        print("Folder to instrument mapping:")
        for folder, instrument in instrument_mapping.items():
            print(f"  {folder} -> {instrument}")
        
        # Initialize stats structure
        instruments = set(instrument_mapping.values())
        for instrument in instruments:
            self.results["instrument_stats"][instrument] = {
                "total_files": 0,
                "total_duration": 0,
                "mean_duration": 0,
                "sessions": {}
            }
            
            # Initialize session tracking 
            for i in range(1, 6):
                session_key = f"ss{i}"
                self.results["instrument_stats"][instrument]["sessions"][session_key] = 0
                
                # Track overall sessions
                if session_key not in self.results["sessions"]:
                    self.results["sessions"][session_key] = 0
        
        # Process each folder
        all_durations = []
        all_amplitudes = []
        all_spectral_centroids = []
        session_counts = {}
        
        for folder in tqdm(instrument_folders, desc="Processing folders"):
            instrument = instrument_mapping[folder]
            folder_path = os.path.join(self.data_path, folder)
            
            # Get all audio files
            audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3'))]
            
            self.results["instrument_stats"][instrument]["total_files"] += len(audio_files)
            
            # Track session information (ss1, ss2, etc.)
            for file in audio_files:
                # Try to identify which session this belongs to
                session = self._identify_session(file)
                if session:
                    self.results["instrument_stats"][instrument]["sessions"][session] += 1
                    self.results["sessions"][session] += 1
                    
                    # Track for plotting
                    if session not in session_counts:
                        session_counts[session] = {}
                    if instrument not in session_counts[session]:
                        session_counts[session][instrument] = 0
                    session_counts[session][instrument] += 1
            
            # Process up to 10 audio files for more detailed analysis
            sample_files = audio_files[:10] if len(audio_files) > 10 else audio_files
            
            for audio_file in tqdm(sample_files, desc=f"Analyzing {instrument} samples", leave=False):
                file_path = os.path.join(folder_path, audio_file)
                
                try:
                    # Load audio with original sample rate
                    y, sr = librosa.load(file_path, sr=None)
                    
                    # Convert to mono if stereo
                    if len(y.shape) > 1:
                        y = librosa.to_mono(y)
                    
                    # Calculate duration
                    duration = len(y) / sr
                    all_durations.append(duration)
                    self.results["instrument_stats"][instrument]["total_duration"] += duration
                    
                    # Calculate amplitude stats
                    amplitude = np.abs(y).max()
                    all_amplitudes.append(amplitude)
                    
                    # Calculate spectral centroid
                    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                    all_spectral_centroids.append(np.mean(cent))
                    
                    # Generate spectrograms for example files (up to 3 per instrument)
                    if len(self.results.get("example_spectrograms", {}).get(instrument, [])) < 3:
                        if "example_spectrograms" not in self.results:
                            self.results["example_spectrograms"] = {}
                        if instrument not in self.results["example_spectrograms"]:
                            self.results["example_spectrograms"][instrument] = []
                        
                        # Create spectrogram
                        spec_filename = f"analysis_results/{instrument}_{len(self.results['example_spectrograms'][instrument])}.png"
                        self._create_spectrogram(y, sr, instrument, audio_file, spec_filename)
                        self.results["example_spectrograms"][instrument].append(spec_filename)
                    
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
        
        # Calculate mean durations
        for instrument in instruments:
            stats = self.results["instrument_stats"][instrument]
            if stats["total_files"] > 0:
                stats["mean_duration"] = stats["total_duration"] / stats["total_files"]
        
        # Save overall stats
        self.results["duration_stats"] = {
            "min": min(all_durations) if all_durations else 0,
            "max": max(all_durations) if all_durations else 0,
            "mean": np.mean(all_durations) if all_durations else 0,
            "median": np.median(all_durations) if all_durations else 0
        }
        
        self.results["amplitude_stats"] = {
            "min": min(all_amplitudes) if all_amplitudes else 0,
            "max": max(all_amplitudes) if all_amplitudes else 0,
            "mean": np.mean(all_amplitudes) if all_amplitudes else 0
        }
        
        self.results["spectral_stats"] = {
            "min_centroid": min(all_spectral_centroids) if all_spectral_centroids else 0,
            "max_centroid": max(all_spectral_centroids) if all_spectral_centroids else 0,
            "mean_centroid": np.mean(all_spectral_centroids) if all_spectral_centroids else 0
        }
        
        # Generate plots
        self._plot_instrument_file_counts()
        self._plot_session_distribution(session_counts)
        self._plot_duration_distributions(all_durations)
        
        # Save the results
        with open("analysis_results/dataset_analysis.json", "w") as f:
            # Convert numpy values to native Python types
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json_data = {k: convert_numpy(v) if not isinstance(v, dict) else 
                         {k2: convert_numpy(v2) for k2, v2 in v.items()} 
                         for k, v in self.results.items()}
            
            json.dump(json_data, f, indent=2)
        
        print(f"Analysis complete. Results saved to analysis_results/dataset_analysis.json")
        return self.results
    
    def _identify_session(self, filename):
        """Identify which session a file belongs to based on naming pattern"""
        filename = filename.lower()
        
        # Look for direct session indicators
        for i in range(1, 6):
            session_key = f"ss{i}"
            if session_key in filename:
                return session_key
        
        # Try to infer from other patterns
        if any(x in filename for x in ["clean", "note", "single"]):
            return "ss1"
        elif any(x in filename for x in ["soft", "loud", "dynamic"]):
            return "ss2"
        elif any(x in filename for x in ["pattern", "play"]):
            return "ss3"
        elif any(x in filename for x in ["position", "record", "loc"]):
            return "ss4"
        elif any(x in filename for x in ["background", "noise", "amb"]):
            return "ss5"
        
        # Fallback - check path for session info
        path = Path(filename)
        parent = path.parent.name.lower()
        for i in range(1, 6):
            session_key = f"ss{i}"
            if session_key in parent:
                return session_key
        
        return None
    
    def _create_spectrogram(self, y, sr, instrument, filename, output_path):
        """Create and save a spectrogram visualization"""
        plt.figure(figsize=(10, 4))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title(f"{instrument} - {filename} (Waveform)")
        
        # Plot spectrogram
        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _plot_instrument_file_counts(self):
        """Plot the file count distribution by instrument"""
        instruments = list(self.results["instrument_stats"].keys())
        file_counts = [self.results["instrument_stats"][i]["total_files"] for i in instruments]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(instruments, file_counts)
        
        # Add count labels on top of bars
        for bar, count in zip(bars, file_counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha='center',
                fontweight='bold'
            )
        
        plt.title('Number of Files per Instrument')
        plt.xlabel('Instrument')
        plt.ylabel('File Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("analysis_results/instrument_file_counts.png")
        plt.close()
    
    def _plot_session_distribution(self, session_counts):
        """Plot the distribution of files across sessions and instruments"""
        # Create a DataFrame-like structure for seaborn
        plot_data = []
        for session, instruments in session_counts.items():
            for instrument, count in instruments.items():
                plot_data.append((session, instrument, count))
        
        # Sort sessions
        sessions = sorted(list(session_counts.keys()))
        instruments = sorted(list(set([p[1] for p in plot_data])))
        
        # Create a matrix for the heatmap
        data_matrix = np.zeros((len(instruments), len(sessions)))
        for session_idx, session in enumerate(sessions):
            for instr_idx, instrument in enumerate(instruments):
                for s, i, count in plot_data:
                    if s == session and i == instrument:
                        data_matrix[instr_idx, session_idx] = count
        
        plt.figure(figsize=(12, 8))
        # Fixed: Changed format to '.1f' to handle floating point values
        sns.heatmap(data_matrix, annot=True, fmt='.0f', cmap='YlGnBu',
                   xticklabels=sessions, yticklabels=instruments)
        plt.title('Distribution of Files Across Sessions and Instruments')
        plt.xlabel('Session')
        plt.ylabel('Instrument')
        plt.tight_layout()
        plt.savefig("analysis_results/session_distribution.png")
        plt.close()
    
    def _plot_duration_distributions(self, durations):
        """Plot the distribution of file durations"""
        plt.figure(figsize=(10, 6))
        plt.hist(durations, bins=20, alpha=0.7)
        plt.axvline(np.mean(durations), color='r', linestyle='--', label=f'Mean: {np.mean(durations):.2f}s')
        plt.axvline(np.median(durations), color='g', linestyle='--', label=f'Median: {np.median(durations):.2f}s')
        plt.title('Distribution of Audio File Durations')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig("analysis_results/duration_distribution.png")
        plt.close()

def main():
    # Change this to your dataset path
    data_path = "dataset"
    
    analyzer = DatasetAnalyzer(data_path)
    results = analyzer.analyze_dataset()
    
    # Print summary of results
    print("\nDataset Summary:")
    print(f"  Total instruments: {len(results['instrument_stats'])}")
    
    total_files = sum(stats["total_files"] for stats in results["instrument_stats"].values())
    print(f"  Total files: {total_files}")
    
    print(f"  Average duration: {results['duration_stats']['mean']:.2f} seconds")
    print(f"  Session distribution:")
    for session, count in results["sessions"].items():
        print(f"    {session}: {count} files")
    
    print("\nInstrument breakdown:")
    for instrument, stats in results["instrument_stats"].items():
        print(f"  {instrument}: {stats['total_files']} files, avg duration: {stats['mean_duration']:.2f}s")
        print(f"    Sessions: {', '.join([f'{s}: {c}' for s, c in stats['sessions'].items() if c > 0])}")

if __name__ == "__main__":
    main()