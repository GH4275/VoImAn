import os
import struct
import numpy as np

def print_dcimg_header(file_path, num_bytes=256):
    """
    Prints a hex dump of the DCIMG file header for raw inspection.
    """
    print(f"--- DCIMG Header (First {num_bytes} bytes) ---")
    with open(file_path, "rb") as f:
        data = f.read(num_bytes)
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_str = ' '.join(f'{b:02X}' for b in chunk)
            ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
            print(f"{i:04X}   {hex_str:<47}  {ascii_str}")
    print("-" * 50)

def get_dcimg_frame_rate(file_path, debug_header=True):
    """
    Parses a Hamamatsu .dcimg file directly to calculate the true average frame rate
    based on internal binary timestamps.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file: {file_path}")

    # Optionally print the raw header bytes
    if debug_header:
        print_dcimg_header(file_path)

    with open(file_path, "rb") as f:
        # Read the file signature to confirm it's a valid DCIMG file
        signature = f.read(8)
        if signature != b"DCIMG\x00\x00\x00":
            raise ValueError("Invalid file format. Not a Hamamatsu .dcimg file.")

        # --- THE FIX ---
        # Offset 36 contains the total number of frames (nfrms) as a 4-byte uint.
        # (Offset 48 actually points to the file size, which caused the inflated frame count!)
        f.seek(36)
        total_frames = struct.unpack("<I", f.read(4))[0]
        
        # Offset 92 typically contains the offset pointing to the timestamp metadata block
        f.seek(92)
        timestamp_offset = struct.unpack("<Q", f.read(8))[0]

        print(f"Total Frames Detected: {total_frames}")

        # Check if the file contains timestamps 
        if timestamp_offset == 0 or total_frames <= 1:
            print("Warning: Insufficient metadata or frames inside the .dcimg binary header.")
            print("Falling back to standard hardware estimations or XML parsing...")
            return None

        # Navigate to the raw timestamp block
        f.seek(timestamp_offset)
        
        # Read the timestamps for all frames 
        # (Using float64/double format common in Hamamatsu's DCAM-API structure)
        raw_timestamps = f.read(total_frames * 8)
        timestamps = np.frombuffer(raw_timestamps, dtype=np.float64)

        # Calculate time differences between successive frames
        time_deltas = np.diff(timestamps)
        
        # Determine average time delta 
        # (Adjusting dynamically if timestamps are logged in milliseconds instead of seconds)
        avg_delta = np.mean(time_deltas)
        if avg_delta > 100:  # Timestamps are likely in milliseconds
            avg_delta /= 1000.0

        # Calculate FPS
        if avg_delta == 0:
            raise ZeroDivisionError("Time delta between frames is zero; cannot calculate FPS.")
            
        fps = 1.0 / avg_delta
        return round(fps, 3)

# --- Example Usage ---
if __name__ == "__main__":
    file_name = r"D:\pAce\BKV009\20260703\FOV1_T1\FOV1_T1_Green.dcimg" 
    
    try:
        calculated_fps = get_dcimg_frame_rate(file_name, debug_header=True)
        if calculated_fps:
            print(f"Calculated Frame Rate: {calculated_fps} FPS")
    except Exception as e:
        print(f"Error parsing .dcimg file: {e}")