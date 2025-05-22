import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import time

if __name__ == '__main__':
    # Load & pisah
    df = pd.read_csv('house-price-india_preprocessed.csv')
    train = df[df['split'] == 'train']
    test  = df[df['split'] == 'test']

    X_train = train.drop(columns=['Price', 'split'])
    y_train = train['Price']
    X_test  = test .drop(columns=['Price', 'split'])
    y_test  = test ['Price']

    # 1. Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid, cv=3, scoring='r2'
    )

    # Ukur durasi training
    t0 = time.time()
    gs.fit(X_train, y_train)
    train_dur = time.time() - t0
    best = gs.best_estimator_

    # Prediksi & metrik manual
    preds = best.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    # Log params & metrics
    mlflow.log_params(gs.best_params_)
    mlflow.log_metric('mse', mse)
    mlflow.log_metric('r2', r2)
    mlflow.log_metric('mae', mae)                  # metrik tambahan #1
    mlflow.log_metric('testing_duration_sec', train_dur)  # metrik tambahan #2

    # Log model
    mlflow.sklearn.log_model(best, 'model')

    print(f"Best params: {gs.best_params_}")
    print(f"MSE: {mse}, R2: {r2}, MAE: {mae}, Time: {train_dur:.2f}s")