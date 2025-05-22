# Age and death parameters
MAX_AGE = 200  # in days
MIN_AGE = 75
GAMMA = 1e-3
DELTA = 0.03454
EXP_SCALE = 1.0

MEAN_AGE_FACTOR = 0.5
STD_AGE_FACTOR = 0.15

HEALTH_NEURON = 1
DEATH_NEURON = -1
DEATH_MIN = -5

GRID_SIZE = 50  # just a spaceing
GRID_SPACING = 0.044  # in mm

k_A = 0.7  # μg/μg/day
k_B = 0.1  # μg/μg/day
k_c = 0.6  # μg/μg/day
D_A = 0.7  # mm^2/day
D_B = 0.5  # mm^2/day

TIME = 250
TIME_SPACING = 0.001
SECONDS = 25  # timesteps
SAVE_INTERVAL = 10
GIF_DURATION = 10

FIG_DPI = 300

SECRETED_VALUE = 4  # μm a day for each μg
PRION_THRESHOLD = 1  # μg per μg neuron # UPDATE NOTES
NUMBER_OF_NEURONS = 500
