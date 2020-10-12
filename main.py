from src.data_management import Settings, MealearningDataHandler
from src.training import training
import numpy as np


def main():
    np.random.seed(999)
    # training_settings = {'method': 'SARIMAX',
    #                      'use_exog': False}

    data_settings = {'dataset': 'sine',
                     'use_exog': False,
                     'training_tasks_pct': 0.75,
                     'validation_tasks_pct': 0.05,
                     'test_tasks_pct': 0.2,
                     'training_points_pct': 0.3,
                     'validation_points_pct': 0.3,
                     'test_points_pct': 0.4,
                     'forecast_length': 6}

    data_settings = Settings(data_settings)
    data = MealearningDataHandler(data_settings)
    #############################################################################
    training_settings = Settings({'method': 'ITL',
                                  'use_exog': False,
                                  'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 2, 36)],
                                  'lags': 6})

    training(data, training_settings)
    #############################################################################
    training_settings = Settings({'method': 'BiasLTL',
                                  'use_exog': False,
                                  'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 2, 36)],
                                  'lags': 6})

    training(data, training_settings)

    k = 1


if __name__ == "__main__":

    main()
