#Construction Noise Detection Using Machine Learning

A end-to-end machine learning project developed for the University of Virginia's Smart & Healthy Buildings course, (CS 4501 - Special Topics in Computer Science) to identify construction-related noise from acoustic sensor data. The goal was to distinguish construction activity from everyday environmental sounds, enabling smarter building monitoring and improving occupant comfort, and using this data to turn on and off a hypothetical white noise actuator.

##Machine Learning Pipeline
* Set up sound sensor and Raspberry Pi to sense and collect sound data for data collection
* Imported, saved, and cleaned and normalized the acoustic sensor data to be of proper formatting
* Split the 24-hour waveform recording into 10-second chunks and manually labeled 1619 data chunks to generate training data
* Built an LSTM-based and random forest classifier-based machine learning model to classify unseen data chunks as having construction noise or not (ended up using the random forest classifier)
* Preprocessed data chunks through feature extraction with Librosa and standard scaling
* Used the machine learning model to predict the existence of construction noise in unseen sound data
* Postprocessed predictions on sound data to find optimal times to turn on and off a hypothetical white noise actuator, balancing factors such as cost, latencies, and disruptions to workflow

##Technologies Used
Python
Pandas
NumPy
Librosa
Scikit-learn
Matplotlib
Jupyter Notebook
Git
GitHub

##Repository Structure
* SAHB Group 7 Full Pipeline.ipynb - full end-to-end pipeline using random forest classifier
* SAHB_Group_7_Data_Processing.py - full end-to-end pipeline using LSTM-based model, but data was not of correct format for this model
* labels_full_with_source.csv - raw output of prediction probabilities and labels for the data chunks
* labels_full_with_source_edit.csv - formatted and edited output from ML model with timestamp, ordered by time
* white_noise_actuator_results.csv - postprocessing metrics ouputs and labels for white noise actuator as YES or NO for each data chunk

**More on the process and details of this project: **
[Adaptive Noise Masking of Construction in Indoor Environments.pdf](https://github.com/user-attachments/files/29361813/Adaptive.Noise.Masking.of.Construction.in.Indoor.Environments.pdf)
