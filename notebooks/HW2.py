import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    from datetime import timedelta
    import matplotlib.pyplot as plt
    import sklearn
    from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import learning_curve
    from sklearn.linear_model import LogisticRegression
    import marimo as mo
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import cross_validate
    import warnings
    warnings.filterwarnings('ignore')
    return (
        DecisionTreeRegressor,
        GradientBoostingRegressor,
        LinearRegression,
        RandomForestRegressor,
        cross_validate,
        mo,
        pd,
        r2_score,
        train_test_split,
    )


@app.cell
def _():
    from sklearn.metrics import mean_absolute_error
    return


@app.cell
def _():
    # pip install scikit-learn
    return


@app.cell
def _(pd):
    cities = pd.read_csv('city.csv')
    print(cities.head())

    calendar = pd.read_csv('Calendar.csv')
    calendar['calendar_date'] = pd.to_datetime(calendar['calendar_date'])
    print(calendar.head())

    passengers = pd.read_csv('passenger.csv')
    passengers['first_call_time'] = pd.to_datetime(passengers['first_call_time'], errors='coerce')

    passengers['date_only'] = passengers['first_call_time'].dt.date
    passengers['date_only'] = pd.to_datetime(passengers['date_only'])
    passengers = passengers.merge(
        calendar[['calendar_date', 'holiday']], 
        left_on='date_only', 
        right_on='calendar_date', 
        how='left'
    )

    trips = pd.read_csv('trip.csv', nrows=10_000)
    # trips.trip_distance.fillna(trips.trip_distance.mean())
    trips['call_time'] = pd.to_datetime(trips['call_time'])
    trips['finish_time'] = pd.to_datetime(trips['finish_time'])

    trips['trip_duration'] = (trips['finish_time'] - trips['call_time']).dt.total_seconds() / 60

    trips['call_hour'] = trips['call_time'].dt.hour
    trips['call_weekday'] = trips['call_time'].dt.weekday
    trips['call_month'] = trips['call_time'].dt.month
    trips['call_date'] = trips['call_time'].dt.date
    trips['call_date'] = pd.to_datetime(trips['call_date'])

    trips__1 = trips.merge(cities, left_on='city_id', right_on='id', how='left', suffixes=('', '_city'))

    trips__1.rename(columns={'name': 'city_name'}, inplace=True)
    trips_2 = trips__1.merge(calendar[['calendar_date', 'holiday', 'week_day']], left_on='call_date', right_on='calendar_date', how='left')

    trips_2['is_weekend'] = trips_2['week_day'].isin(['Sunday', 'Saturday']).astype(int)
    trips_2['speed'] = trips_2['trip_distance'] / trips_2['trip_duration']
    city_counts = trips_2['id_city'].value_counts()
    data = trips_2.copy()
    day_mapping = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }

    data['week_day']=data['week_day'].map(day_mapping)
    return (data,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():

    # data['base_fare'] = 2.5 + data['trip_distance'] * 1.8 + data['trip_duration'] * 0.25
    # data['peak_surcharge'] = ((data['call_hour'] >= 7) & (data['call_hour'] <= 9)).astype(int) * 3 + \
    #                          ((data['call_hour'] >= 17) & (data['call_hour'] <= 19)).astype(int) * 4
    # data['weekend_surcharge'] = data['is_weekend'] * 2
    # data['surge_multiplier'] = 1 + data['surge_rate']
    # data['total_fare'] = (data['base_fare'] + data['peak_surcharge'] + data['weekend_surcharge']) * data['surge_multiplier']
    # data['total_fare'] = np.maximum(data['total_fare'], 5)
    # data['profitability'] = data['total_fare'] / (data['trip_distance'] * 0.8 + data['trip_duration'] * 0.1 + 2)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(data, mo):

    period = mo.ui.dropdown(
        options=["Current month", "Last 7 days", "Last 30 days", "All"],
        value="Current month",
        label="Period"
    )

    cities_ = ["All"] + sorted(data['city_name'].dropna().unique().tolist())
    city = mo.ui.dropdown(
        options=cities_, 
        value="All", 
        label="City"
    )

    weekdays = ["All", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_filter = mo.ui.dropdown(
        options=weekdays,
        value="All",
        label="Week Day"
    )

    hour_filter = mo.ui.dropdown(
        options=["All", "(6-12)", "(12-18)", "(18-24)", "(0-6)"],
        value="All",
        label="Time of the day"
    )

    max_distance = int(data['trip_distance'].max())
    min_distance = mo.ui.slider(
        start=0,
        stop=max_distance,
        value=0,
        label=f" Min distance (км) [0-{max_distance}]"
    )

    weekend_only = mo.ui.checkbox(
        label="Only weekends",
        value=False
    )

    fare_range = mo.ui.range_slider(
        start=0,
        stop=int(data['trip_fare'].max()) + 10,
        value=[0, int(data['trip_fare'].max())],
        label="Price"
    )

    return (
        city,
        hour_filter,
        min_distance,
        period,
        weekday_filter,
        weekend_only,
    )


@app.cell
def _(
    city,
    data,
    hour_filter,
    min_distance,
    mo,
    period,
    weekday_filter,
    weekend_only,
):
    def calculate_and_show_kpi():
        period_val = period.value
        city_val = city.value
        weekday_val = weekday_filter.value
        hour_range_val = hour_filter.value
        min_dist_val = min_distance.value
        weekend_val = weekend_only.value

        df = data.copy()

        if city_val != "All":
            df = df[df['city_name'] == city_val]

        if weekday_val != "All":
            weekday_map = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2, 
                "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
            }
            df = df[df['call_weekday'] == weekday_map[weekday_val]]

        if hour_range_val != "All":
            if hour_range_val == "(6-12)":
                df = df[(df['call_hour'] >= 6) & (df['call_hour'] < 12)]
            elif hour_range_val == "(12-18)":
                df = df[(df['call_hour'] >= 12) & (df['call_hour'] < 18)]
            elif hour_range_val == "(18-24)":
                df = df[(df['call_hour'] >= 18) & (df['call_hour'] < 24)]
            elif hour_range_val == "(0-6)":
                df = df[(df['call_hour'] >= 0) & (df['call_hour'] < 6)]

        if min_dist_val > 0:
            df = df[df['trip_distance'] >= min_dist_val]


        if weekend_val:
            df = df[df['is_weekend'] == True]

        if len(df) == 0:
            return mo.callout("No data")

        total_rides = len(df)
        avg_fare = df['trip_fare'].mean()
        total_revenue = df['trip_fare'].sum()
        avg_distance = df['trip_distance'].mean()
        avg_duration = df['trip_duration'].mean()

        return mo.vstack([
            mo.md("## KPI:"),
            mo.hstack([
                mo.stat(label="Trips", value=f"{total_rides:,}", caption=period_val),
                mo.stat(label="Fare", value=f"${total_revenue:,.0f}", caption="all"),
                mo.stat(label="Avg fare", value=f"${avg_fare:.2f}", caption="per trip"),
                mo.stat(label="Distance", value=f"{avg_distance:.1f} km", caption="Avg"),
            ], gap=2),
            mo.md(f"**Filters:** {city_val}, {period_val}")
        ], gap=2)

    mo.vstack([
        mo.md("### Filters:"),
        mo.hstack([period, city, weekday_filter, hour_filter], gap=2),
        mo.hstack([min_distance,  weekend_only], gap=3),
        mo.md("---"),
        calculate_and_show_kpi()
    ], gap=2)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(data):
    data_=data.fillna(0)
    return (data_,)


@app.cell
def _(data_, train_test_split):
    X = data_[['trip_distance', 'trip_duration', 'surge_rate', 'is_weekend', 'call_hour']]
    y = data_['trip_fare']

    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    model_choice = mo.ui.dropdown(
        options=['LinearRegression', 'DecisionTree', 'RandomForest', 'GradientBoosting'],
        value='LinearRegression',
        label='Model:'
    )

    train_counter = mo.ui.number(
        value=0,
        label="Counter"
    )

    train_btn = mo.ui.button(value=0,on_click=lambda value: value + 1, label="Train!")
    return model_choice, train_btn, train_counter


@app.cell
def _():
    return


@app.cell
def _(
    DecisionTreeRegressor,
    GradientBoostingRegressor,
    LinearRegression,
    RandomForestRegressor,
    X_test,
    X_train,
    cross_validate,
    mo,
    model_choice,
    r2_score,
    train_btn,
    train_counter,
    y_test,
    y_train,
):
    if train_counter.value:
        train_counter.value = train_counter.value + 1

    def train_simple_model():
        if model_choice.value == 'LinearRegression':
            model = LinearRegression()
        elif model_choice.value == 'DecisionTree':
            model = DecisionTreeRegressor(max_depth=5)
        elif model_choice.value == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100)
        else:
            model = GradientBoostingRegressor(n_estimators=100)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        r2 = r2_score(y_train, y_pred)
        r2_test = r2_score(y_test, y_pred_test)
    
        scoring = {
            'r2': 'r2',
            'neg_mean_squared_error': 'neg_mean_squared_error',
            'neg_mean_absolute_error': 'neg_mean_absolute_error'
        }
    
        cv_results = cross_validate(model, X_train, y_train, 
                                    cv=5, scoring=scoring, 
                                    return_train_score=False)
    
        cv_r2_mean = cv_results['test_r2'].mean()
        cv_r2_std = cv_results['test_r2'].std()
        cv_mse = -cv_results['test_neg_mean_squared_error'].mean()
        cv_mae = -cv_results['test_neg_mean_absolute_error'].mean()

        return mo.vstack([
            mo.md(f"## Model: **{model_choice.value}**"),
            mo.md("### Main metrics"),
            mo.hstack([
                mo.stat(label="R² train", value=f"{r2:.3f}", caption="Training"),
                mo.stat(label="R² test", value=f"{r2_test:.3f}", caption="Test"),
                mo.stat(label="CV R²", value=f"{cv_r2_mean:.3f}", caption=f"±{cv_r2_std:.3f}"),
            ], gap=2),
            mo.md("### Cross-validation details"),
            mo.hstack([
                mo.stat(label="CV MSE", value=f"{cv_mse:.3f}", caption="Mean Squared Error"),
                mo.stat(label="CV MAE", value=f"{cv_mae:.3f}", caption="Mean Absolute Error"),
            ], gap=2),
            mo.md(f"**R² scores per fold:** {', '.join([f'{score:.3f}' for score in cv_results['test_r2']])}"),
        ], gap=2)

    mo.vstack([
        mo.md("# Training panel"),
        mo.hstack([model_choice,train_btn], gap=2),
        mo.md("---"),

        train_simple_model() if train_btn.value > 0 else mo.callout(
            "Choose the model and train it",
            kind="info"
        )
    ], gap=3)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
