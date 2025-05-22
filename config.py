# Simulation configuration parameters

# Age and neuron death parameters
MAX_AGE = 200  # Maximum neuron age (days)
MIN_AGE = 75  # Minimum neuron age (days)
GAMMA = 1e-3  # Death rate parameter
DELTA = 0.03454  # Death rate parameter
EXP_SCALE = 1.0  # Exponential scaling factor

MEAN_AGE_FACTOR = 0.5  # Mean age factor for neuron initialization
STD_AGE_FACTOR = 0.15  # Standard deviation factor for neuron age

HEALTH_NEURON = 1  # Value representing a healthy neuron
DEATH_NEURON = -1  # Value representing a dead neuron
DEATH_MIN = -5  # Minimum value for neuron death state

NEW_CELL_CHANGE = 0.7  # Change in cell state for new cells

# Grid and diffusion parameters
GRID_SIZE = 50  # Grid size (number of cells per side)
GRID_SPACING = 0.045  # Grid spacing (mm)

k_A = 0.7  # Protein decay rate (μg/μg/day)
k_B = 0.01  # Prion decay rate (μg/μg/day)
k_c = 1.2  # Conversion rate (μg/μg/day)
D_A = 0.7  # Protein diffusion coefficient (mm^2/day)
D_B = 0.6  # Prion diffusion coefficient (mm^2/day)

# Simulation time parameters
TIME = 300  # Total simulation time (days)
TIME_SPACING = 0.01  # Time step size (days * spacing)
SECONDS = 25  # Number of timesteps per second (for animation)
SAVE_INTERVAL = 10  # Interval for saving simulation state (timesteps)
GIF_DURATION = 1000  # Duration of GIF animations (seconds)

# Plotting parameters
FIG_DPI = 300  # Figure resolution

# Biological parameters
SECRETED_VALUE = 0.4  # Protein secreted per neuron per day (μm/μg)
PRION_THRESHOLD = 0.001  # Prion threshold for neuron death (μg/μg neuron)
PRION_GROWTH_THRESHOLD = 0.1  # Prion growth rate (μg/μg/day)
NUMBER_OF_NEURONS = 500  # Total number of neurons in the simulation
