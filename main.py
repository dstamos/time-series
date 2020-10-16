from src.data_management import Settings, MealearningDataHandler
from src.training import training
import numpy as np


def main():
    np.random.seed(999)
    data_settings = {'dataset': 'sine',
                     'use_exog': False,
                     'training_tasks_pct': 0.7,
                     'validation_tasks_pct': 0.1,
                     'test_tasks_pct': 0.2,
                     'training_points_pct': 0.5,
                     'validation_points_pct': 0.1,
                     'test_points_pct': 0.4,
                     'forecast_length': 1}

    data_settings = Settings(data_settings)
    data = MealearningDataHandler(data_settings)
    #############################################################################
    # training_settings = Settings({'method': 'ITL',
    #                               'use_exog': False,
    #                               'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 2, 16)],
    #                               'lags': 3})
    # #
    # model_itl = training(data, training_settings)
    #############################################################################
    # training_settings = Settings({'method': 'BiasLTL',
    #                               'use_exog': False,
    #                               'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 2, 16)],
    #                               'lags': 3})
    # #
    # model_ltl = training(data, training_settings)
    #############################################################################
    # training_settings = Settings({'method': 'SARIMAX',
    #                               'use_exog': False})
    #
    # model_sarimax = training(data, training_settings)
    #############################################################################
    training_settings = Settings({'method': 'xgboost',
                                  'use_exog': False,
                                  'lags': 3})

    model_xgboost = training(data, training_settings)
    # TODO
    """
    SARIMAX
    xgboost
    
    benchmark
    """

    k = 1


if __name__ == "__main__":

    main()
