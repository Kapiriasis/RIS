import os
from src.channel_model import ChannelModel
from src.ris_model import RISModel
from src.simulation import Simulation
from src.utils import load_input_parameters, plot_results

def _find_input_parameters_path():
    # Resolve path to input_parameters.json from cwd or repo layout.
    candidates = [
        'data/input_parameters.json',  # cwd = repo root
        os.path.join(os.path.dirname(__file__), '..', 'data', 'input_parameters.json'),  # cwd = ris-wireless-simulation
    ]
    for path in candidates:
        p = os.path.normpath(os.path.abspath(path))
        if os.path.isfile(p):
            return p
    return candidates[0]  # let load_input_parameters create or report error

def main():
    input_params = load_input_parameters(_find_input_parameters_path())
    if input_params is None:
        print("Failed to load input parameters. Exiting.")
        return

    channel_model = ChannelModel()
    ris_model = RISModel()
    ris_model.configure_from_params(input_params)

    simulation = Simulation(channel_model, ris_model)
    results = simulation.run_simulation(input_params)
    simulation.analyze_results(results)
    plot_results(results)

if __name__ == "__main__":
    main()
