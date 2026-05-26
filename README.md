**F1 Track Analysis — DSP Course Project**
This is my personal project developed alongside the Digital Signal Processing (DSP) course at the University of Graz (2026). The project applies DSP concepts to Formula 1 lap telemetry data — treating race car sensor signals (speed, throttle, brake, gear) as real-world signals to analyse using the techniques taught in the course.
The app is built with Streamlit and developed entirely on my local machine, then published here as a portfolio of the practical work I completed during the course.

**What This Project Is About**
Formula 1 cars generate thousands of data points per second from onboard sensors. I used this telemetry data as the raw signal source for each DSP exercise — making the theory tangible by applying it to something real and visual. Each exercise corresponds to a topic from the DSP course and demonstrates that concept applied to an actual F1 lap.

**Project Structure**

f1_track_analysis/
├── Home.py              # Landing page — exercise overview and navigation
├── pages/
│   ├── 1_Time_Domain.py         # Exercise 1 — Time Domain Analysis
│   ├── 2_Frequency_Domain.py    # Exercise 2 — Frequency Domain Analysis
│   └── 3_Time_Frequency.py      # Exercise 3 — Time-Frequency Analysis
├── data/
│   └── *.csv                    # Lap telemetry data (tracked via Git LFS)
├── .gitattributes               # Git LFS config for CSV files
├── requirements.txt
└── README.md

**Exercise Descriptions**
Exercise 1 — Time Domain Analysis ✅
Analyses raw F1 lap telemetry signals directly in the time domain. The lap is treated as a time series where each channel (speed, throttle position, brake pressure, steering angle) is plotted against time or distance along the track.
Key topics covered:
Signal sampling and discrete-time representation
Signal amplitude, mean, variance, and energy
Identifying key events in the signal (braking zones, acceleration phases, cornering)
Comparing two drivers' signals in the same time domain to highlight differences in driving style
Exercise 2 — Frequency Domain Analysis ✅
Transforms the telemetry signals from the time domain into the frequency domain using the Fast Fourier Transform (FFT). This reveals which frequency components dominate the signal — for example, engine vibration patterns, suspension oscillation, or periodic braking rhythms around a circuit.
Key topics covered:
Discrete Fourier Transform (DFT) and FFT algorithm
Magnitude and phase spectrum
Power Spectral Density (PSD)
Identifying dominant frequencies in speed and vibration signals
Filtering interpretation — which frequencies carry meaningful information vs. noise
Exercise 3 — Time-Frequency Analysis ✅
Because F1 signals are non-stationary (the car is constantly changing speed and behaviour), a pure frequency domain view loses the time information. This exercise applies time-frequency analysis to see how the frequency content of the signal changes over the course of a lap.
Key topics covered:
Short-Time Fourier Transform (STFT) and spectrograms
Window functions and their effect on time-frequency resolution
Wavelet transform as an alternative to STFT
Visualising how engine RPM, gear changes, and braking events appear in the time-frequency plane
Resolution trade-off: time resolution vs. frequency resolution

**Dependencies**
streamlit
fastf1
pandas
numpy
scipy
matplotlib
plotly

**Course Context**
Course: Digital Signal Processing
Institution: FH Joanneum
Year: 2026
Student: Leni Pazhayakariyil Iype
This project was built concurrently with the course  each exercise was developed and tested locally as the corresponding theory was covered in lectures. The goal was to go beyond abstract examples and see DSP principles in action on a real, high-frequency engineering dataset.
