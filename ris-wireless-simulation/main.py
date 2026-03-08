from src.channel_model import ChannelModel
from src.ris_model import RISModel
from src.simulation import Simulation
from src.utils import load_input_parameters, plot_results

def main():
    # Load input parameters
    input_params = load_input_parameters('data/input_parameters.json')
    if input_params is None:
        print("Failed to load input parameters. Exiting.")
        return

    # Initialize channel model
    channel_model = ChannelModel()

    # Initialize RIS model
    ris_model = RISModel()
    ris_model.set_num_elements(input_params['num_elements'])
    ris_model.set_placement(input_params['placement'])

    # Run simulation
    simulation = Simulation(channel_model, ris_model)
    results = simulation.run_simulation(input_params)

    # Analyze results
    simulation.analyze_results(results)

    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main()
