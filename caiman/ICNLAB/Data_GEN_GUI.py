#Runs TrainingDataGenerator.py
#Run using run_Data_GEN_GUI.bat

import tkinter as tk
from tkinter import messagebox
import os
import threading
from tkinterdnd2 import DND_FILES, TkinterDnD
import subprocess

SCRIPT_PATH = r"C:\Users\ICNLab\CaImAn_GV\caiman\ICNLAB\TrainingDataGenerator.py"
folder_paths = []

import re

# ---------- Functions ----------
def remove_selected():
    selected = folder_listbox.curselection()
    for index in reversed(selected):
        folder_listbox.delete(index)

def clear_list():
    folder_listbox.delete(0, tk.END)

def drop(event):
    base_paths = root.tk.splitlist(event.data)
    date_re = re.compile(r'^\d{8}$')
    fov_re = re.compile(r'^FOV\d+_T\d+$')
    for base_path in base_paths:
        if os.path.isdir(base_path) and base_path not in folder_listbox.get(0, tk.END):
            for root, dirs, files in os.walk(base_path):
                parts = os.path.normpath(root).split(os.sep)

                # need at least: base / anything / yyyymmdd / FOV#_T#
                if len(parts) < 3:
                    continue

                if date_re.match(parts[-2]) and fov_re.match(parts[-1]):
                    folder_paths.append(root)

            folder_listbox.insert(tk.END, folder_paths)

def run_caiman_thread():
    folders = folder_listbox.get(0, tk.END)
    if not folders:
        messagebox.showerror("Error", "No folders selected.")
        return

    root_folder = os.path.dirname(folders[0])
    log_file_path = os.path.join(root_folder, "processed_folders_DATAGEN.txt")

    with open(log_file_path, "w") as log_file:
        log_file.write("Processed Mouse ID Folders:\n")

    for i, folder in enumerate(folders, start=1):
        if not os.path.isdir(folder):
            messagebox.showwarning("Warning", f"Skipping invalid folder: {folder}")
            continue

        try:
            cmd = f'cmd.exe /k python "{SCRIPT_PATH}" "{folder}"'
            process = subprocess.Popen(cmd)
            process.wait()

            with open(log_file_path, "a") as log_file:
                log_file.write(f"{folder}\n")

        except Exception as e:
            messagebox.showerror("Launch error", str(e))
            break

    messagebox.showinfo("Done", f"Processing complete.\nLog saved at:\n{log_file_path}")

def run_caiman():
    threading.Thread(target=run_caiman_thread, daemon=True).start()

# ---------- GUI ----------
root = TkinterDnD.Tk()
root.title("TRAINING DATA GENERATOR")
root.geometry("700x450")

tk.Label(root, text="Drag and drop MAIN FOLDER/Drive for DATAGEN:").pack(pady=(10,0))

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

# Run button
tk.Button(root, text="Process Data for Mouse ID", bg="#4CAF50", fg="white",
          height=3, command=run_caiman).pack(pady=15)

root.mainloop()






# import tkinter as tk
# from tkinter import filedialog, messagebox
# import subprocess
# import os

# SCRIPT_PATH = r"C:\Users\ICNLab\CaImAn_GV\caiman\ICNLAB\test_single_trial_RAM_DISK_5.4_simple.py"
# DEFAULT_DATA_DIR = r"C:\Users\ICNLab\caiman_data\testdata\testdata"


# def select_folder():
#     folder = filedialog.askdirectory(initialdir=DEFAULT_DATA_DIR)
#     if folder:
#         folder_var.set(folder)


# def run_caiman():
#     folder = folder_var.get()

#     if not os.path.isdir(folder):
#         messagebox.showerror("Error", "Please select a valid data folder.")
#         return

#     if not os.path.isfile(SCRIPT_PATH):
#         messagebox.showerror("Error", f"Script not found:\n{SCRIPT_PATH}")
#         return

#     cmd = f'cmd.exe /k python "{SCRIPT_PATH}" "{folder}"'

#     try:
#         subprocess.Popen(cmd)
#     except Exception as e:
#         messagebox.showerror("Launch error", str(e))


# # ---------- GUI ----------
# root = tk.Tk()
# root.title("VoImAn Pipeline Runner")   # <-- CHANGED
# root.geometry("600x180")

# folder_var = tk.StringVar(value=DEFAULT_DATA_DIR)

# tk.Label(root, text="Select Mouse ID Folder:").pack(pady=(15, 5))

# frame = tk.Frame(root)
# frame.pack(fill="x", padx=10)

# tk.Entry(frame, textvariable=folder_var).pack(side="left", fill="x", expand=True)
# tk.Button(frame, text="Browse", command=select_folder).pack(side="left", padx=5)

# tk.Button(
#     root,
#     text="Process Data for Mouse ID",  # <-- CHANGED
#     command=run_caiman,
#     height=2,
#     bg="#4CAF50",
#     fg="white"
# ).pack(pady=20)

# root.mainloop()
