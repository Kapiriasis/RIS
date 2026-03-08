import numpy as np

class RISModel:
    def __init__(self):
        self.element_size = 0.5  # Default
        self.placement = [0, 0, 5]  # Default position
        self.num_elements = 100
        self.phases = np.zeros(self.num_elements)  # Phase shifts

    def set_element_size(self, size):
        self.element_size = size

    def set_placement(self, placement):
        self.placement = placement

    def set_num_elements(self, num):
        self.num_elements = num
        self.phases = np.zeros(num)

    def set_phase_reflections(self, phases):
        # self.phases = np.array(phases)
        self.phases = np.random.uniform(0, 2 * np.pi, self.num_elements)

    def calculate_reflection_coefficient(self, incident_angle, reflected_angle):
        # Simplified reflection with phase
        # Assuming perfect reflection with adjustable phase
        reflection_coeff = np.exp(1j * self.phases)
        return reflection_coeff
