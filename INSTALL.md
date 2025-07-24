# 🛠️ Installation Guide

This document contains setup instructions for the project environment.

---

## 📦 Requirements

- Python 3.10+
- pip
- virtualenv (optional, but recommended)

---

## 🧰 Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   
## 🛠 Installation

Follow these steps to set up the project locally.  

### 1. Install Miniconda

Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### 2. Install Build Tools

Depending on your OS:

- **Windows:**  
  Download and install the **Microsoft Visual C++ Build Tools**:  
  [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)  
  During installation, select **"Desktop Development with C++"**.

- **Linux:**  
  Install the GCC compiler:  
  ```bash
  sudo apt install build-essential
  ```

- **MacOS:**
  In MacOS the equivalent of GCC is already mostly covered by:
  ```bash
  xcode-select --install
  ```

### 3. Set Up the Conda Environment
Open Anaconda Prompt (Windows) or a terminal (Linux) and navigate to the project directory:

- **Windows:**
  ```bash
  cd \path\to\repo
  py -3.8 -m venv py38
  .\py38\Scripts\activate
  ```
- **Linux & MacOS:**
  ```bash
  cd /path/to/repo
  conda create -n py38 python=3.8
  conda activate py38
  ```

### 4. Install Required Packages

```bash
# follow this order
pip install -e /path/to/gtfs_railways/external_packages/osmread
pip install -e /path/to/gtfs_railways/external_packages/gtfspy
pip install -e /path/to/gtfs_railways
```
