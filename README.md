# Face Recognition System – Installation & Setup Guide

## 1️⃣ Install Visual C++ Build Tools (Required for `insightface`)

Download the installer:

👉 [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Run the `.exe` file and select:

### Workload

✔ **Desktop development with C++**

---

### Included (Make sure these are selected)

* C++ Build Tools core features
* Visual C++ v14 Redistributable Update
* C++ core desktop features

---

### Optional (Check the following)

* MSVC Build Tools for x64/x86 (Latest)
* Windows 11 SDK (10.0.26100.7705)
* C++ CMake tools for Windows
* Testing tools core features – Build Tools
* MSVC AddressSanitizer
* vcpkg package manager
* MSVC v143 – VS 2022 C++ x64/x86 build tools

Then complete the installation.

---

# 2️⃣ Create Project Environment

### Create an empty project folder

```bash
mkdir face-recognition-project
cd face-recognition-project
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

# 3️⃣ Install Required Python Packages

Install packages in **exact order**:

### Step 1 – Upgrade pip tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

### Step 2 – Core Scientific Libraries

```bash
pip install numpy==1.24.4 scipy==1.15.3 matplotlib==3.10.8 sympy==1.14.0
```

---

### Step 3 – Face Recognition Dependencies

```bash
pip install insightface==0.7.3 onnx==1.20.1 onnxruntime==1.16.3 opencv-python==4.9.0.80 scikit-image==0.25.2 scikit-learn==1.3.2 albumentations==2.0.8
```

---

### Step 4 – Image Processing Libraries

```bash
pip install pillow==10.2.0 imageio==2.37.2 tifffile==2025.5.10
```

---

### Step 5 – Web Server Dependencies

```bash
pip install flask==2.3.3 waitress==3.0.2
```

---

### Step 6 – Utility Libraries

```bash
pip install tqdm==4.67.3 prettytable==3.17.0 easydict==1.13 requests==2.32.5 pyyaml==6.0.3 cython==3.2.4
```

---

### Step 7 – Reinstall Numpy (Important Fix)

```bash
pip install numpy==1.24.4 --force-reinstall
```

---

# 4️⃣ Run the Application

```bash
python app.py
```

Wait 5–6 seconds.

Open in browser:

```
http://127.0.0.1:5000/
```

---

# 5️⃣ How to Use the Application

## 🏠 Home Page

* Camera will automatically start.

---

## ➕ Add User

1. Click **Add User** from the navigation bar.
2. On the right side under **👤 Person Information**, type the person's name.
3. Click **Start Capture**.
4. System captures **50 images** (0/50 → 50/50).

---

## 🧠 Generate Embeddings

1. Click **🧠 Generate Embeddings** from navbar.
2. Wait 30 seconds to 2 minutes (depends on dataset size).
3. This converts captured images into recognition embeddings.

---

## 🔙 Return to Recognition

Click **Back to Recognition** at the top.

---

## 🎯 Command Control

Use voice or manual command input.

Example:

```
Detect the person right of User1
Detect the person left of User1
```

* Detection is based on camera orientation.
* If no person is found → red warning text appears.
* If detected → verification snapshot popup appears.

---

## 🎨 Generate Sketch Diagram

1. Click **Generate Sketch Diagram**
2. Click OK on alert
3. Scroll down to view the generated sketch

---

# ✅ Setup Complete
