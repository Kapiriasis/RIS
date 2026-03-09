import os
from src.channel_model import ChannelModel
from src.ris_model import RISModel
from src.simulation import Simulation
from src.utils import load_input_parameters, plot_results

def _find_input_parameters_path():
    """Resolve path to input_parameters.json from cwd or repo layout."""
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
    ris_model.set_num_elements(input_params['num_elements'])
    ris_model.set_placement(input_params['placement'])
    ris_model.set_layout(input_params.get('layout', '2d'))
    ris_model.set_reflection_amplitude(input_params.get('reflection_amplitude', 0.9))
    ris_model.set_phase_quantization_bits(input_params.get('phase_quantization_bits', 2))
    ris_model.set_phase_error_std(input_params.get('phase_error_std', 0.0))

    simulation = Simulation(channel_model, ris_model)
    results = simulation.run_simulation(input_params)
    simulation.analyze_results(results)
    plot_results(results)

if __name__ == "__main__":
    main()
