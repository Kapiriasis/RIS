# Simulation and Performance Analysis of Reconfigurable Intelligent Surfaces for Next-Generation Wireless Networks

Reconfigurable Intelligent Surfaces (RIS) are a key enabler for 6G wireless systems, capable of intelligently controlling the propagation of electromagnetic waves to enhance coverage, spectral efficiency and energy efficiency. This project will model and simulate RIS-assisted wireless communication systems, focusing on how RIS can improve link quality compared to traditional wireless channels. The project will involve implementing RIS-aided channel models in MATLAB/Python, simulating different RIS configurations and analyzing system performance in terms of SNR, capacity and energy efficiency.

## Project structure

```
thesis_code/
├── data/
│   └── params.json
├── results/
├── scripts/
│   ├── ber.py
│   ├── network.py
│   ├── direct.py
│   ├── relay.py
│   └── ris.py
├── src/
│   ├── channel.py
│   ├── datagen.py
│   ├── handover.py
│   ├── plot.py
│   ├── ris_channel.py
│   ├── ris_element.py
│   ├── signal.py
│   ├── user.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running 

Run main.py

Plots are saved under `results/`

## Configuration

Edit `config/params.json`
 