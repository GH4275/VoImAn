# **VoImAn (CaImAn Fork) Installation & Troubleshooting Guide**

* **Target OS:** Windows 10 / 11  
* **Installation Type:** Developer Mode (Source-linked)

This document outlines the complete procedure for installing the **VoImAn** suite (a custom fork of **CaImAn**). Developer mode installation allows users to modify source files directly without needing to reinstall the package. Because scientific Python environments involve complex dependencies (e.g., TensorFlow, OpenCV), please follow these instructions carefully and sequentially.

## **Table of Contents**

1. [Phase 1: System Prerequisites](#bookmark=id.nfzauzv0796w)  
2. [Phase 2: Repository and Environment Setup](#bookmark=id.v98rux2fqndp)  
3. [Phase 3: Developer Mode Build](#bookmark=id.kcr2xpeomemh)  
4. [Phase 4: Data Initialization and Dependencies](#bookmark=id.rau5s3oek6w2)  
5. [Phase 5: Fetching Required Models](#bookmark=id.4wc1y3blivdt)  
6. [Phase 6: RAM Disk Setup](#bookmark=id.3a7ef1faumpb)  
7. [Phase 7: Troubleshooting Common Errors](#bookmark=id.dxhrbpfjx80z)

## **Phase 1: System Prerequisites**

Before beginning the Conda environment setup, your Windows machine must be prepared to handle C/C++ compilation and deeply nested file structures.

### **1\. Install Anaconda & Mamba (If Not Already Installed)**

If this is a brand-new computer or you do not have a Python environment manager installed, you will need to install Anaconda (or Miniconda) and the Mamba package manager before proceeding.

#### **Step A: Install Anaconda (or Miniconda)**

*Miniconda is generally recommended for developer setups as it is much lighter, but either will work perfectly.*

1. Download the Windows installer for [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download).  
2. Run the installer.  
3. **Important:** When prompted, select **"Just Me"** for the installation type, and **do not** check the box to add Anaconda to your system PATH (this prevents conflicts with other software).  
4. Once installed, open your Windows Start menu and search for **Anaconda Prompt** (or Miniconda Prompt). You will use this specific terminal for all remaining steps.

#### **Step B: Install Mamba**

*While Conda is the default package manager, this installation requires Mamba, which is a heavily optimized, C++ based version of Conda that resolves complex dependencies much faster.*

1. Open your **Anaconda Prompt** (or Miniconda Prompt).  
2. Run the following command to install Mamba into your base environment:

conda install \-n base \-c conda-forge mamba \-y

### **2\. Enable Long File Paths (Crucial for TensorFlow)**

Windows natively caps file paths at 260 characters, which will cause the TensorFlow installation cache to corrupt and fail. You must disable this limit.

1. Click the Windows **Start** button, type PowerShell.  
2. Right-click **Windows PowerShell** and select **Run as Administrator**.  
3. Execute the following command:

New-ItemProperty \-Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" \-Name "LongPathsEnabled" \-Value 1 \-PropertyType DWORD \-Force

4. **Recommendation:** Restart your computer after applying this change.

💡 **Alternative Recommendation (No Admin Privileges):** \> If you cannot acquire administrator privileges, you can bypass this issue by moving the Conda cache location to a shorter path:

mkdir C:\\Users\\YOUR\_USER\\cpkgs   
conda config \--add pkgs\_dirs C:\\Users\\YOUR\_USER\\cpkgs

### **3\. Install Microsoft Build Tools**

CaImAn requires C/C++ compilers to build its extensions.

1. Open a standard Windows Command Prompt or PowerShell.  
2. Run the following winget command:

winget install \--id=Microsoft.VisualStudio.2019.BuildTools \-e

*(If already installed, the installer will notify you).*

### **4\. Adjust Virtual Memory**

If your machine has less than 64GB of physical RAM, the original CaImAn installation guide recommends increasing the maximum size of your Windows pagefile to 64GB or more. The Windows memmap interface is sensitive to memory limits, and leaving it at the default can cause memory-mapping errors when processing large imaging datasets.

## **Phase 2: Repository and Environment Setup**

*All subsequent steps must be executed from an **Anaconda Prompt** or **Miniconda Prompt** (do not use standard PowerShell).*

### **1\. Clone the Repository**

Download the source code to your local machine:

git clone git@github.com:GH4275/VoImAn.git  
cd VoImAn

⚠️ **Note:** If you receive an SSH Permission denied (publickey) error, refer to the [Troubleshooting](#bookmark=id.dxhrbpfjx80z) section or clone using HTTPS instead:

git clone https://github.com/GH4275/VoImAn.git  
cd VoImAn

### **2\. Create the Conda Environment**

We use mamba to resolve the environment.yml dependencies efficiently.

mamba env create \-f environment.yml \-n caiman

📝 **Note on the environment file:** Custom modifications have been made to the environment.yml file to ensure compatibility on Windows. Specifically, tkinterdnd2 and tensorflow==2.10.0 are handled via pip to bypass Conda-forge indexing limitations and Windows path issues.

## **Phase 3: Developer Mode Build**

Once the environment is successfully created, you must compile the C/C++ extensions and link the environment to your local folder.

1. Activate the environment:

conda activate caiman

2. Link the Visual Studio 2019 compiler variables to the environment:

mamba install \-n caiman vs2019\_win-64

3. Build and install the package in "editable" (developer) mode:

pip install \-e .

## **Phase 4: Data Initialization and Dependencies**

### **1\. The pkg\_resources Patch**

Modern versions of setuptools (v82.0+) have removed the pkg\_resources module, which CaImAn's internal manager requires to locate files. You must install a legacy version of setuptools before initializing the data folders.

mamba install "setuptools\<82"

### **2\. Initialize the Working Directory**

You must set up a working directory for code samples and datasets. Because you installed in developer mode, use the \--inplace flag. This links the data folder to your current repository rather than downloading fresh copies.

caimanmanager install \--inplace

By default, this creates a caiman\_data folder in your Windows Home directory (C:\\Users\\\<YourUsername\>\\caiman\_data).

💡 **Optional configuration:** If you prefer to store this data elsewhere, set the environment variable prior to running the install command:

set CAIMAN\_DATA=C:\\Path\\To\\Your\\Preferred\\Folder  
caimanmanager install \--inplace

## **Phase 5: Fetching Required Models**

The suite requires a pre-trained neural network model. Due to recent Google Drive anti-scraping policies, automated download scripts (like download\_model.py using gdown) will likely fail with a FileURLRetrievalError.

### **Manual Download Instructions**

1. Try running the download script:

python caiman\\ICNLAB\\download\_model.py

2. Open your web browser and navigate to the Google Drive link outputted by the script error statement:

[Google Drive Model Link](https://drive.google.com/file/d/1ZPdQqhW6-V1bh6v30sRrWBAmf3ciRvlm/view?usp=drive_link)

3. Download the model file manually.  
4. Place the downloaded model file in the following repository folder:

.../VoImAn/caiman/ICNLAB/

## **Phase 6: RAM Disk Setup**

The final step to get everything up and running is to set up a virtual disk using the computer's RAM. This is critical for running the analysis of high-frequency datasets (up to 35-second recordings at 640 Hz).

This setup uses a pre-existing toolkit called **ImDiskToolkit**, which is included as a zip archive inside your repository.

1. Locate and open the ImDiskTk-x64.zip file found in the .../VoImAn/caiman/ICNLAB/ folder using Windows File Explorer.  
2. Navigate inside the extracted folder and double-click the Install.bat file to run the installer.  
3. In the installer options, select settings to **install for all users** and **create desktop shortcuts**.  
4. Once installed, run the **RamDisk UI** application.  
5. Configure and create a **64 GB** drive with the letter **R:**.  
6. Check the option to **dynamically allocate memory**. Keep all other default settings.  
7. Close the application. You can verify the creation of the new virtual drive letter by checking **This PC** in Windows File Explorer.

*That’s it\! You have completed the core installation.*

## **Phase 7: Troubleshooting Common Errors**

### **❌ Error: libmamba Package cache error or Invalid package cache**

* **Cause:** Windows blocked Mamba from downloading a package (usually tensorflow) because the file path exceeded 260 characters.  
* **Solution:** 1\. Complete **Phase 1, Step 2** of this guide (Enable Long File Paths in Registry).  
  2\. Navigate to C:\\Users\\\<YourUsername\>\\AppData\\Local\\miniconda3\\pkgs.  
  3\. Delete the partially downloaded tensorflow folder/package files.  
  4\. Rerun the environment creation command:

mamba env create \-f environment.yml \-n caiman

### **❌ Error: gdown.exceptions.FileURLRetrievalError: Cannot retrieve the public link...**

* **Cause:** Google Drive blocks automated tools from downloading files to prevent bandwidth scraping.  
* **Solution:** Refer to **Phase 5**. You must download the model file manually via a web browser using the provided link and place it in the designated path.

### **❌ Error: Permission denied (publickey) during Git Clone**

* **Cause:** Your local machine does not have an SSH key registered with GitHub.  
* **Solution:** Generate an SSH key or fall back to HTTPS cloning:  
  * **Option A (HTTPS Fallback):** Clone using HTTPS instead:

git clone https://github.com/GH4275/VoImAn.git

* **Option B (Generate SSH):** Generate an SSH key and add the public key to your GitHub account settings:

ssh-keygen \-t ed25519 \-C "your\_email@example.com"  
