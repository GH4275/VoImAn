import tkinter as tk
from tkinter import messagebox
import os
import threading
from tkinterdnd2 import DND_FILES, TkinterDnD
import subprocess

# ---------- Functions ----------
def remove_selected():
    selected = folder_listbox.curselection()
    for index in reversed(selected):
        folder_listbox.delete(index)

def clear_list():
    folder_listbox.delete(0, tk.END)

def drop(event):
    paths = root.tk.splitlist(event.data)
    for path in paths:
        # If a text file is dropped, read the relative paths inside it
        if os.path.isfile(path) and path.endswith('.txt'):
            with open(path, 'r') as file:
                for line in file:
                    cleaned_path = line.strip()
                    if cleaned_path and cleaned_path not in folder_listbox.get(0, tk.END):
                        folder_listbox.insert(tk.END, cleaned_path)
        
        elif os.path.isdir(path) and path not in folder_listbox.get(0, tk.END):
            folder_listbox.insert(tk.END, path)

def run_caiman_thread():
    # 1. Check for folders
    folders = folder_listbox.get(0, tk.END)
    if not folders:
        messagebox.showerror("Error", "No folders selected.")
        return
    
    # 2. Check for Frame Rate
    fr = fr_entry.get().strip()
    if not fr:
        messagebox.showwarning("Warning", "Please provide a frame rate before running.")
        return
    
    root_folder = os.path.dirname(folders[0])

    # Get the selected analysis mode
    mode = analysis_mode.get()

    # Filter valid folders
    valid_folders = [f for f in folders if os.path.isdir(f)]

    if not valid_folders:
        messagebox.showinfo("Info", "No valid folders to process.")
        return

    # Pass mode as arg 2, fr as arg 3, and then the folders
    cmd = (
        'cmd.exe /k python "run_trials_controller.py" '
        f'{mode} {fr} ' +
        ' '.join(f'"{f}"' for f in valid_folders)
    )

    subprocess.Popen(cmd)

def run_caiman():
    threading.Thread(target=run_caiman_thread, daemon=True).start()

# ---------- GUI ----------
root = TkinterDnD.Tk()
root.title("VoImAn Pipeline Runner")
root.geometry("700x500") # Slightly increased height to fit the new textbox
analysis_mode = tk.StringVar(value="new")  # default

tk.Label(root, text="Drag and drop Mouse ID or Trial folders below:").pack(pady=(10,0))

# Listbox for folders
folder_listbox = tk.Listbox(root, selectmode=tk.EXTENDED, width=80, height=15)
folder_listbox.pack(padx=10, pady=10, fill="both", expand=True)
folder_listbox.drop_target_register(DND_FILES)
folder_listbox.dnd_bind('<<Drop>>', drop)

# Buttons frame
button_frame = tk.Frame(root)
button_frame.pack(pady=5)
tk.Button(button_frame, text="Remove Selected", width=20, command=remove_selected).pack(side="left", padx=5)
tk.Button(button_frame, text="Clear List", width=20, command=clear_list).pack(side="left", padx=5)

# Analysis mode selection
mode_frame = tk.LabelFrame(root, text="Analysis Mode")
mode_frame.pack(pady=10)

tk.Radiobutton(mode_frame, text="New", value="new",
               variable=analysis_mode).pack(side="left", padx=10)

tk.Radiobutton(mode_frame, text="Old", value="old",
               variable=analysis_mode).pack(side="left", padx=10)

tk.Radiobutton(mode_frame, text="All", value="all",
               variable=analysis_mode).pack(side="left", padx=10)

# Frame Rate Input
fr_frame = tk.Frame(root)
fr_frame.pack(pady=5)
tk.Label(fr_frame, text="Frame Rate (Hz):").pack(side="left")
fr_entry = tk.Entry(fr_frame, width=10)
fr_entry.pack(side="left", padx=5)

# Run button
tk.Button(root, text="Process Data for Mouse ID", bg="#4CAF50", fg="white",
          height=3, command=run_caiman).pack(pady=15)

root.mainloop()