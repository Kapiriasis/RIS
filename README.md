# Simulation and Performance Analysis of Reconfigurable Intelligent Surfaces for Next-Generation Wireless Networks

Reconfigurable Intelligent Surfaces (RIS) are a key enabler for 6G wireless systems, capable of intelligently controlling the propagation of electromagnetic waves to enhance coverage, spectral efficiency and energy efficiency. This project will model and simulate RIS-assisted wireless communication systems, focusing on how RIS can improve link quality compared to traditional wireless channels. The project will involve implementing RIS-aided channel models in MATLAB/Python, simulating different RIS configurations and analyzing system performance in terms of SNR, capacity and energy efficiency.

## Project structure

```
thesis_code/
├── config/
│   └── params.json
├── src/
│   ├── channel.py
│   └── utils.py
├── scripts/
│   ├── run_base.py
|   └── run_relay.py
├── results/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running 



Plots are saved under `results/`

## Configuration

Edit `config/params.json`
