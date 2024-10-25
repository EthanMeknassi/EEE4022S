# Target Motion Reconstruction using FMCW radar for Hand gestures 

## Author
**Ethan Meknassi**

## Introduction
For a detailed understanding of the project, please refer to the final thesis report entitled: `Hand-gesture recognition using FMCW radar and Machine Learning Thesis`

## Overview
This repository contains a set of files and tools for processing radar data. The primary files and folders are as follows:

### 1. `pipelineSave.py`
- **Description:** This script runs the entire data processing pipeline and saves the resulting maps.
- **Functionality:** It is currently configured to save maps every 50 frames, which is consistent with the structure of the HDF5 files used in this project. The HDF5 files are organized so that each file contains data for one hand gesture performed every 50 frames. For example, a 7500-frame HDF5 file would have 150 sample gestures.

### 2. `processData.py`
- **Description:** This script is responsible for the core data processing tasks.
- **Functionality:** It takes raw radar data in the form of an HDF5 file and processes it into maps. This includes generating Range-Doppler Maps (RDM), applying windowing techniques, and implementing clutter removal methods. The output of this process is a Range-Time Map (RTM) and a Doppler-Time Map (DTM).

### 3. `Initialization Functions`
- **Description:** This folder contains the functions required for running both `pipelineSave.py` and `processData.py`.
- **Contents:** Functions in this folder are essential for constructing the data cube from frames and generating IQ data. These functions are based on code originally developed by William Bourn. Although not all functions in this folder are used in this application, they are accurate and potentially useful for other applications.

### 4. `Processed Data`
- **Description:** This folder contains sample processed data that was utilized in the implementation of this project.
- **Data:** The processed data includes five different hand gestures: grabbing, lifting, pulling, pushing, and patting.

### 5. `combineImage.py`
- **Description:** This script is designed to combine the processed data and output a combined image of the two maps (RTM and DTM).

### 6. `Radar Config File`
- **Description:** This folder contains the radar configuration file, which details the parameters used for configuring the radar in this implementation.

## Important Note
Please be aware that you need to readjust the file paths in each script to make the system usable in your specific setup.

Feel free to explore this repository and use the provided tools for your radar data processing needs. If you have any questions or encounter issues, please don't hesitate to contact the author, Ethan Meknassi at:

https://www.linkedin.com/in/ethan-meknassi888/
