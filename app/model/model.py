import dill as pickle
from pathlib import Path

__version__ = '0.1.0'

BASE_DIR = Path(__file__).resolve(strict=True).parent
print(BASE_DIR)



with open(f'{BASE_DIR}/random_forest_classifier-{__version__}.pkl', 'rb') as file:
    model = pickle.load(file)
def predict(x):
    return model.predict(