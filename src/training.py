from src.ltl import BiasLTL
from src.sarimax import Sarimax
from src.xgboost import Xgboost
from src.independent_learning import ITL


def training(data, training_settings):
    method = training_settings.method

    if method == 'ITL':
        model = ITL(training_settings)
        model.fit(data.test_tasks)
    elif method == 'BiasLTL':
        model = BiasLTL(training_settings)
        model.fit(data.training_tasks, data.validation_tasks)
        model.predict(data.test_tasks)
    elif method == 'SARIMAX':
        model = Sarimax(training_settings)
        model.fit(data.test_tasks)
        model.predict(data.test_tasks)
    elif method == 'xgboost':
        model = Xgboost(training_settings)
        model.fit(data.test_tasks)
    else:
        raise ValueError('Unknown method', method)
    return model
