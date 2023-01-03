import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import datetime
from datetime import datetime
import holidays
import time
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import iplot

start_time = time.time()

#TO DO:
#Need to implement the following models: RNN, RETAIN, MINA
#Need to web crawl for indicators to include in input_data
#Need to add in indicators into input data as new columns


def plot_dataset(input_data, title):
    #A function used to plot the input dataset
    data = []
    value = go.Scatter(
        x = input_data.index,
        y = input_data.Open,
        mode = "lines",
        name = "values",
        marker = dict(),
        text = input_data.index,
        line = dict(color = "rgba(50,0,0, 0.3)"),
    )
    data.append(value)

    layout = dict(
        title = title,
        xaxis = dict(title="Date", ticklen=5, zeroline=False),
        yaxis = dict(title="Open", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    #Fix later. Not vital, but helpful #####################################################################
    fig.show()

def logistic_regression(input_data):
    train_size = 0.75
    trainindex = int(len(input_data) * train_size)

    best_aucrocs = []
    for run in range(20):
        print("Run: ", run)
        train_data = input_data[:trainindex]
        test_data = input_data[trainindex:]

        model = linear_model.LogisticRegression(max_iter = 500)
        model.fit(train_data[['Open', 'High', "Low"]], train_data['Adj Close Feat'])
        predict_probabilities = np.array([a[1] for a in model.predict_proba(test_data[['Open', 'High', 'Low']])])
        best_aucrocs.append(roc_auc_score(test_data['Adj Close Feat'], predict_probabilities))

    print("Average AUCROC: ", np.round(np.mean(best_aucrocs), 4), "+/-", np.round(np.std(best_aucrocs), 4))
    #make a combine dataframe of test_data X, Y and predictions. Then make two graphs to compare the differences
    #Between the Y and the prediction. ########################################################################
    minutes = int(time.time() - start_time)
    seconds = int(((time.time() - start_time) - minutes) * 60)
    print("Logistic regression completed in : ", int(minutes/60), " minutes and ", seconds, "seconds.")

class RNNModel(nn.Module): 
    #This is a basic RNN implementation. As this project is in progress, for more fine tuned and advanced RNNs
    #please look to other projects on my github (Learning latent spaces and Time Series Data Algorithms).
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden_state = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad()
        out, hidden_state = self.rnn(x, hidden_state.cuda().detach())
        out = out[:, -1:]
        out = self.fc(out)
        return out

class LSTMModel(nn.Module):
    #This is a basic LSTM implementation. As this project is in progress, for more fine tuned and advanced RNNs
    #please look to other projects on my github (Learning latent spaces and Time Series Data Algorithms).
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.cuda().detach(), c0.cuda().detach()))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    #This is a basic GRU implementation. As this project is in progress, for more fine tuned and advanced RNNs
    #please look to other projects on my github (Learning latent spaces and Time Series Data Algorithms).
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.gru(x, h0.cuda().detach())
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class RETAIN(): #################################################################################
    pass

class MINA(): #? Might work CNN + RNN for binary classification #################################
    pass

class Optimization:
    #We set up our optimizations in this step and prepare our chosen model for training, and evaluation.
    #We can optionally save our model at the end to enable future use for new, unclassified data.
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        self.model.train()
        yhat = self.model(x)
        loss = self.loss_fn(y, yhat)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        model_path = f'{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

#        torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().cpu().numpy())
                values.append(y_test.to(device).detach().cpu().numpy())

        return predictions, values

    def plot_losses(self): ###################################################################################
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

def NN_prep(input_data):
    #Used to format the data into train and test group batches
    batch_size = 8
    train_size = 0.75
    trainindex = int(len(input_data) * train_size)

    train_loader = DataLoader(dataset = input_data[:trainindex], batch_size = batch_size)
    test_loader = DataLoader(dataset = input_data[trainindex:], batch_size = batch_size)
    return train_loader, test_loader

def generate_timelag(input_data, n_lags):
    #Uses historical data to create a time series for each time interval of the data
    open_data = input_data.copy()
    for n in range(1, n_lags + 1):
        open_data[f"t{n}"] = open_data["Open"].shift(n)
    open_data = open_data.iloc[n_lags:]
    return open_data

def one_hot_encoding(input_data, cols):
    #Converts the data into a one hot encoding data table
    for col in cols:
        dummies = pd.get_dummies(input_data[col], prefix=col)
    return pd.concat([input_data, dummies], axis = 1).drop(columns=cols)

def cyclical_features(input_data, col_name, period, start_num=0):
    #Converts the linear data variables (day, month, year etc...) into a cyclical representation
    #Which aids in the algorithms understanding of concept of a year being a cycle vs a linear 1 - 12,
    #then returning back to 1 again.
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(input_data[col_name] - start_num) / period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(input_data[col_name] - start_num) / period)
    }
    print(input_data)
    return input_data.assign(**kwargs).drop(columns=[col_name])

def is_holiday(date):
    #Create a list of holidays as a feature
    date = date.replace(hour = 0)
    return 1 if (date in holidays.US) else 0

def add_holiday_col(input_data, holidays):
    #Apply the holiday as a feature
    return input_data.assign(is_holiday = input_data.index.to_series().apply(is_holiday))

def feature_label_split(input_data, target_col):
    #Separate the labels from our data
    y = input_data[[target_col]]
    X = input_data.drop(columns=[target_col])
    return X, y

def train_val_test_split(input_data, target_col, test_ratio):
    #Separate our data into training batches, validation batches and test batches
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(input_data, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, shuffle = False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_ratio, shuffle = False)
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_scaler(scaler):
    #Experiment with different scalar types by returning the called scaler operation
    scalers = {
        "1": MinMaxScaler,
        "2": StandardScaler,
        "3": MaxAbsScaler,
        "4": RobustScaler,
    }
    return scalers.get(scaler.lower())()

def get_model(model, model_params):
    #Experiment with different implemented models by returning the called NN.
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)

def inverse_transform(scaler, input_data, columns):
    #Transform our input data into the inverse format
    for col in columns:
        input_data[col] = scaler.inverse_transform(input_data[col])
    return input_data

def format_prediction(predictions, values, input_test, scaler):
    vals = np.concatenate(values, axis = 0).ravel()
    preds = np.concatenate(predictions, axis = 0).ravel()
    result = pd.DataFrame(data = {"value": vals, "prediction": preds}, index = input_test.head(len(vals)).index)
    result = result.sort_index()
    result = inverse_transform(scaler, result, [["value", "prediction"]])
    return result

def calculate_metrics(input_data):
    #Experiment with different evaluation metrics to observe overfitting in a variety of ways.
    return {
        'mae': mean_absolute_error(input_data.value, input_data.prediction),
        'rmse': mean_squared_error(input_data.value, input_data.prediction) ** 0.5,
        'r2': r2_score(input_data.value, input_data.prediction)
    }

def plot_predictions(df_result):
    #Plot our predictions
    data = []
    
    value = go.Scatter(
        x=df_result.index,
        y=df_result.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df_result.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)
    
    prediction = go.Scatter(
        x=df_result.index,
        y=df_result.prediction,
        mode="lines",
        line={"dash": "dot"},
        name='predictions',
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(prediction)
    
    layout = dict(
        title="Predictions vs Actual Values for the dataset",
        xaxis=dict(title="Time", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)


pd.set_option('display.max_columns', None)
path = "~/Documents/Stock algorithm/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"{device}" " is available.")
input_data = pd.read_csv(path + 'CADUSD_daily.csv')
input_data["Open"] = input_data["Open"].shift(1)
input_data.drop(input_data.head(1).index, inplace = True)
input_data['Adj Close Feat'] = [1 if ele > 0 else 0 for ele in (input_data['Open'] - input_data['Adj Close'])]
input_data = input_data.drop(columns = ["Close", "Adj Close Feat"])

input_data = input_data.dropna()

print(input_data)

input_data = input_data.set_index(['Date'])
input_data.index = pd.to_datetime(input_data.index)
if not input_data.index.is_monotonic_increasing:
    input_data = input_data.sort_index()

n_lags = 25
batch_size = 64
output_dim = 1
hidden_dim = 64
layer_dim = 3
dropout = 0.15
n_epochs = 200
learning_rate = 0.00082
weight_decay = 0.0001
runs = 1

#plot_dataset(input_data, title="") ############################################################

#logistic regression
#logistic_regression(input_data)

#Record first time and print here ##############################################################

#train_loader, test_loader = NN_prep(input_data)
##Generate features for data for attention mechanisms
time_lag_data = generate_timelag(input_data, n_lags)

##Feature engineering
features = (
            #Consider removing day and hour to follow pattern ##################################
            #.assign(minute_of_hour = time_lag_data.index.minuteofhour)
            #.assign(hour_of_day = time_lage_data.index.hourofday)
            time_lag_data
            .assign(day_of_week = time_lag_data.index.dayofweek)
            .assign(week_of_year = time_lag_data.index.week)
)

#features = one_hot_encoding(features, ['week_of_year'])

#features = cyclical_features(input_data, 'minute_of_hour', 60, 0)
#features = cyclical_features(input_data, 'hour_of_day', 24, 0)
features = cyclical_features(features, 'day_of_week', 5, 0)
features = cyclical_features(features, 'week_of_year', 52, 0)

#features = add_holiday_col(features)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(features, 'Adj Close', 0.2)

print(X_train)

##Applying scaler
for i in range(4):
    #runs each type of scaler
    print(i+1)
    scaler = get_scaler(str(i+1))
    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train)
    y_val_arr = scaler.transform(y_val)
    y_test_arr = scaler.transform(y_test)

    train_features = torch.Tensor(X_train_arr)
    train_targets = torch.Tensor(y_train_arr)
    val_features = torch.Tensor(X_val_arr)
    val_targets = torch.Tensor(y_val_arr)
    test_features = torch.Tensor(X_test_arr)
    test_targets = torch.Tensor(y_test_arr)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)

    #Training
    input_dim = len(X_train.columns)
    model_params = {'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'layer_dim': layer_dim,
                'output_dim': output_dim,
                'dropout_prob': dropout}

    #for i in range models, run each model

    for run in range(runs):
        #Runs multiple models
        model = get_model('gru', model_params)
        model = model.cuda()
        loss_fn = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        opt = Optimization(model = model, loss_fn = loss_fn, optimizer = optimizer)
        opt.train(train_loader, val_loader, batch_size = batch_size, n_epochs = n_epochs, n_features = input_dim)
        opt.plot_losses()

        prediction, values = opt.evaluate(test_loader, batch_size = batch_size, n_features = input_dim)
        result = format_prediction(prediction, values, X_test, scaler)
        result_metrics = calculate_metrics(result)

        for key, value in result_metrics.items():
            print(key,'\t',  value)

        plot_predictions(result)
        print(time.time() - start_time)