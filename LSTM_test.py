import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
import keras.backend as K
import math as ma

# def root_mean_squared_error(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true)))

AllStation = []
Stationpath = "D://code_sets//ds_bigproject//data_set//AllStation.txt"
with open(Stationpath , "r") as f:
    sline = f.readlines()

for line in sline :
    AllStation.append(str(line.rstrip()))
    
mode = int(input("input mode(single 1 or all station 2) : "))

### INPUT MODE TO CHOOSE TEST ONLY ONE STATION OR TEST FOR ALL STATION
if (mode == 1):

### SINGLE STATION CODE
## START

#translate whole raw data be a array
    name = str(input("input your station :"))
    backforward = int(input("back day : "))
    data_common_path = "D://code_sets//ds_bigproject//data_set//train_data//"
    with open(data_common_path+name+".txt" , "r") as f:
        lines = f.readlines()

    data_arr = []

    for line in lines:
        dataline = [float(x) for x in line.strip().split()]
        data_arr.append(dataline)

    data_arr = data_arr[::-1]

    data_arr = np.array(data_arr)
    feature_sets = data_arr[:,:6]
    rainfall_sets = data_arr[:,6]

    data_size = len(data_arr)

    # Generate synthetic data, assuming 100 time steps with 6 features each
    # X_train represents the input features of the training data, Y_train represents the corresponding rainfall
    X_train = feature_sets[:int(data_size/2) , :]
    Y_train = rainfall_sets[:int(data_size/2) ]

    # Reshape the input data, adding a time step dimension
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

    # Create the model
    model = Sequential()

    # Add the LSTM layer, units represent the number of LSTM units, input_shape represents the input data shape (timesteps, features)
    model.add(LSTM(units=50, input_shape=(1, 6)))

    # Add the output layer, which outputs a single value representing the predicted rainfall
    model.add(Dense(units=1))

    # Compile the model, selecting the loss function and optimizer
    # model.compile(optimizer='adam', loss='mse')
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    count_class_0 = (Y_train < 0.5).sum()
    count_class_1 = (Y_train >= 0.5).sum()
    
    # calculate weight
    weight_class_0 = 1
    weight_class_1 = 1 + 2 *count_class_0 / (count_class_0 + count_class_1)
    
    
    # weight_class_0 = 1 / (count_class_0 )
    # weight_class_1 = 1 / (count_class_1)

    # setting weight
    class_weight = {0: weight_class_0, 1: weight_class_1}
    
    # model.fit(X_train, Y_train, epochs=10, batch_size=32)
    model.fit(X_train, Y_train, epochs=10*10, batch_size=32 , class_weight = class_weight)

    # Generate synthetic test data
    # X_test = feature_sets[int(data_size/2):data_size,:]
    # Y_test = rainfall_sets[int(data_size/2):data_size]
    X_test = feature_sets[data_size-backforward*24-1:data_size,:]
    Y_test = rainfall_sets[data_size-backforward*24-1:data_size]
    #int(data_size/2):data_size

    # Reshape the test data, adding a time step dimension
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Use the model to make predictions
    predicted_rainfall = model.predict(X_test)
    #predicted_rainfall = model.predict(np.ones(len(X_test)))

    #Test : print the prediction
    # for i in range(len(Y_test)):
    #     print(str(Y_test[i]) + " " + str(predicted_rainfall[i]))
    #print(max(predicted_rainfall))
    # print(len(predicted_rainfall))

    # Calculate errors
    #errors = np.abs(predicted_rainfall - Y_test)
    error = 0
    acrate = 0
    ac_sunny = 0
    ac_rain = 0
    num_rain = 0
    num_sunny = 0
    errors = []
    for i in range(len(Y_test)):
        errors.append(abs((Y_test[i] - predicted_rainfall[i])))
        #errors.append(abs(Y_test[i] - predicted_rainfall[i]))
        error = error + abs((Y_test[i] - predicted_rainfall[i])*(Y_test[i] - predicted_rainfall[i]))
        if (Y_test[i] >= 0.5):
            num_rain += 1
        else:
            num_sunny += 1
        if ( (Y_test[i] >= 0.5 and predicted_rainfall[i] >= 0.5) or (Y_test[i] < 0.5 and predicted_rainfall[i] < 0.5)):
            acrate = acrate + 1
        if (Y_test[i] >= 0.5 and predicted_rainfall[i] >= 0.5):
            ac_rain += 1
        if (Y_test[i] < 0.5 and predicted_rainfall[i] < 0.5):
            ac_sunny += 1
    errors = np.array(errors)

    #Test : print the error
    error = np.sqrt(error[0]/len(Y_test))
    print("the RMSE is : "+str(error))
    acrate = acrate / len(Y_test)
    acrainrate = ac_rain / num_rain
    acsunnyrate = ac_sunny / num_sunny
    print("the AC Rate is : " + str(acrate))
    print("the ac rain is : "+ str(acrainrate))
    print("the ac sunny rate is : "+ str(acsunnyrate))

    print("predict max rainfall : " + str(max(predicted_rainfall)[0]))

    # for i in range(len(errors)):
    #     print(errors[i])

    # Plot the actual vs predicted rainfall
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.plot(Y_test, label='Actual Rainfall')
    #plt.plot(predicted_rainfall, label='Predicted Rainfall')
    plt.plot(0.5*np.ones(len(Y_test)), label='rain line', color='red')
    plt.xlabel('Time')
    plt.ylabel('Rainfall')
    plt.title('Actual Rainfall')
    plt.legend()

    # Plot the absolute errors
    plt.subplot(2, 2, 2)
    plt.plot(predicted_rainfall, label='Predicted Rainfall', color='orange')
    plt.plot(0.5*np.ones(len(Y_test)), label='rain line', color='red')
    plt.xlabel('Time')
    plt.ylabel('Rainfall')
    plt.title('Predicted Rainfall')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(Y_test, label='Actual Rainfall')
    plt.plot(predicted_rainfall, label='Predicted Rainfall')
    plt.plot(0.5*np.ones(len(Y_test)), label='rain line', color='red')
    plt.xlabel('Time')
    plt.ylabel('Rainfall')
    plt.title('Actual vs Predicted Rainfall')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(error*np.ones(len(Y_test)), label='error')
    plt.plot(acrate*np.ones(len(Y_test)), label='AC rate')
    plt.plot(acrainrate*np.ones(len(Y_test)), label='AC rain rate')
    plt.plot(acsunnyrate*np.ones(len(Y_test)), label='AC sunny rate')
    plt.xlabel('Time')
    plt.ylabel('value')
    plt.title('Information')
    plt.legend()

    plt.suptitle(name)

    #plt.savefig("D://code_sets//ds_bigproject//data_set//output_image//"+"rainfall_prediction"+name + ".png")  # Save the plot as an image
    plt.show()

## END
### SINGLE STATION CODE

elif (mode == 2):
### ALL STATION CODE
### START
    backforward = int(input("back day : "))
    for name in AllStation:
        # name = str(input("input your station :"))
        data_common_path = "D://code_sets//ds_bigproject//data_set//train_data//"
        with open(data_common_path+name+".txt" , "r") as f:
            lines = f.readlines()

        data_arr = []

        for line in lines:
            dataline = [float(x) for x in line.strip().split()]
            data_arr.append(dataline)

        data_arr = data_arr[::-1]

        data_arr = np.array(data_arr)
        feature_sets = data_arr[:,:6]
        rainfall_sets = data_arr[:,6]

        data_size = len(data_arr)

        # Generate synthetic data, assuming 100 time steps with 6 features each
        # X_train represents the input features of the training data, Y_train represents the corresponding rainfall
        X_train = feature_sets[:int(data_size/2) , :]
        Y_train = rainfall_sets[:int(data_size/2) ]

        # Reshape the input data, adding a time step dimension
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

        # Create the model
        model = Sequential()

        # Add the LSTM layer, units represent the number of LSTM units, input_shape represents the input data shape (timesteps, features)
        model.add(LSTM(units=50, input_shape=(1, 6)))

        # Add the output layer, which outputs a single value representing the predicted rainfall
        model.add(Dense(units=1))

        # Compile the model, selecting the loss function and optimizer
        #model.compile(optimizer='adam', loss='mse')
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        count_class_0 = (Y_train < 0.5).sum()
        count_class_1 = (Y_train >= 0.5).sum()
        
        # calculate weight
        weight_class_0 = count_class_1 / (count_class_0 + count_class_1)
        weight_class_1 = count_class_0 / (count_class_0 + count_class_1)

        # setting weight
        class_weight = {0: weight_class_0, 1: weight_class_1}
        
        # model.fit(X_train, Y_train, epochs=10, batch_size=32)
        model.fit(X_train, Y_train, epochs=10*10, batch_size=32 , class_weight = class_weight)

        # Generate synthetic test data
        # X_test = feature_sets[int(data_size/2):data_size,:]
        # Y_test = rainfall_sets[int(data_size/2):data_size]
        X_test = feature_sets[data_size-backforward*24-1:data_size,:]
        Y_test = rainfall_sets[data_size-backforward*24-1:data_size]   
        #int(data_size/2):data_size

        # Reshape the test data, adding a time step dimension
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        # Use the model to make predictions
        predicted_rainfall = model.predict(X_test)
        #predicted_rainfall = model.predict(np.ones(len(X_test)))

        #Test : print the prediction
        # for i in range(len(Y_test)):
        #     print(str(Y_test[i]) + " " + str(predicted_rainfall[i]))
        #print(max(predicted_rainfall))
        # print(len(predicted_rainfall))

        # Calculate errors
        #errors = np.abs(predicted_rainfall - Y_test)
        error = 0
        acrate = 0
        ac_sunny = 0
        ac_rain = 0
        num_rain = 0
        num_sunny = 0
        errors = []
        for i in range(len(Y_test)):
            errors.append(abs((Y_test[i] - predicted_rainfall[i])))
            #errors.append(abs(Y_test[i] - predicted_rainfall[i]))
            error = error + abs((Y_test[i] - predicted_rainfall[i])*(Y_test[i] - predicted_rainfall[i]))
            if (Y_test[i] >= 0.5):
                num_rain += 1
            else:
                num_sunny += 1
            if ( (Y_test[i] >= 0.5 and predicted_rainfall[i] >= 0.5) or (Y_test[i] < 0.5 and predicted_rainfall[i] < 0.5)):
                acrate = acrate + 1
            if (Y_test[i] >= 0.5 and predicted_rainfall[i] >= 0.5):
                ac_rain += 1
            if (Y_test[i] < 0.5 and predicted_rainfall[i] < 0.5):
                ac_sunny += 1
        errors = np.array(errors)

        #Test : print the error
        error = np.sqrt(error[0]/len(Y_test))
        print("the RMSE is : "+str(error))
        acrate = acrate / len(Y_test)
        acrainrate = ac_rain / num_rain
        acsunnyrate = ac_sunny / num_sunny
        print("the AC Rate is : " + str(acrate))
        print("the ac rain is : "+ str(acrainrate))
        print("the ac sunny rate is : "+ str(acsunnyrate))

        print("predict max rainfall : " + str(max(predicted_rainfall)[0]))

        # for i in range(len(errors)):
        #     print(errors[i])

        # Plot the actual vs predicted rainfall
        plt.figure(figsize=(10, 5))

        plt.subplot(2, 2, 1)
        plt.plot(Y_test, label='Actual Rainfall')
        #plt.plot(predicted_rainfall, label='Predicted Rainfall')
        plt.plot(0.5*np.ones(len(Y_test)), label='rain line', color='red')
        plt.xlabel('Time')
        plt.ylabel('Rainfall')
        plt.title('Actual Rainfall')
        plt.legend()

        # Plot the absolute errors
        plt.subplot(2, 2, 2)
        plt.plot(predicted_rainfall, label='Predicted Rainfall', color='orange')
        plt.plot(0.5*np.ones(len(Y_test)), label='rain line', color='red')
        plt.xlabel('Time')
        plt.ylabel('Rainfall')
        plt.title('Predicted Rainfall')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(Y_test, label='Actual Rainfall')
        plt.plot(predicted_rainfall, label='Predicted Rainfall')
        plt.plot(0.5*np.ones(len(Y_test)), label='rain line', color='red')
        plt.xlabel('Time')
        plt.ylabel('Rainfall')
        plt.title('Actual vs Predicted Rainfall')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(error*np.ones(len(Y_test)), label='error')
        plt.plot(acrate*np.ones(len(Y_test)), label='AC rate')
        plt.plot(acrainrate*np.ones(len(Y_test)), label='AC rain rate')
        plt.plot(acsunnyrate*np.ones(len(Y_test)), label='AC sunny rate')
        plt.xlabel('Time')
        plt.ylabel('value')
        plt.title('Information')
        plt.legend()

        plt.suptitle(name)

        plt.savefig("D://code_sets//ds_bigproject//data_set//output_image//"+"rainfall_prediction"+name + ".png")  # Save the plot as an image
    #plt.show()
else:
    print("mode is not supported")

### END
### ALL STATION CODE
