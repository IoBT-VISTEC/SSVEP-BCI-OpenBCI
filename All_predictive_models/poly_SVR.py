#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io
import numpy as np
import itertools
from scipy.signal import butter, lfilter
from scipy.fftpack import fft, fftfreq
from random import randint
import scipy.signal
import pandas as pd
import time
from math import sqrt
from pprint import pprint
from functools import reduce
from matplotlib import pyplot as plt
from IPython.display import display, Markdown
import math
import pylab
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn import metrics, preprocessing 
from sklearn.svm import SVC , SVR
from sklearn import neighbors
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import xlwt
from xlrd import open_workbook
from openpyxl import load_workbook
import xlrd
import xlutils
from xlutils.copy import copy
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


# ## Define Function

# In[2]:


def my_r2_score(v_true, v_pred):
    ssres = np.sum(np.square(v_true - v_pred))
    sstot = np.sum(np.square(v_true - np.mean(v_true)))
    return 1 - ssres / sstot
def curve_fitting(data):
    if len(data.shape) == 3:
        target_data = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
        time_seq = np.linspace(1, int(data.shape[2]), int(data.shape[2]))
        #Data was used
        for p in range(data.shape[0]):
            for c in range(data.shape[1]):
                if c == 0:
                    X_data = data[p, c, :]
                    poly_features = PolynomialFeatures(degree = 1)
                    X_poly = poly_features.fit_transform(time_seq.reshape(-1,1))
                    poly_model = LinearRegression() 
                    poly_model.fit(X_poly, X_data)
                    pred = poly_model.predict(X_poly)
                    target_data[p, c, :] = pred
                else:
                    X_data = data[p, c, :]
                    poly_features = PolynomialFeatures(degree = 2)
                    X_poly = poly_features.fit_transform(time_seq.reshape(-1,1))
                    poly_model = LinearRegression() 
                    poly_model.fit(X_poly, X_data)
                    pred = poly_model.predict(X_poly)
                    target_data[p, c, :] = pred
    else:
        target_data = np.zeros((data.shape[0], data.shape[1]))
        time_seq = np.linspace(1, int(data.shape[1]), int(data.shape[1]))
        #Data was used
        for c in range(data.shape[0]):
            if c == 0:
                X_data = data[c, :]
                poly_features = PolynomialFeatures(degree = 1)
                X_poly = poly_features.fit_transform(time_seq.reshape(-1,1))
                poly_model = LinearRegression() 
                poly_model.fit(X_poly, X_data)
                pred = poly_model.predict(X_poly)
                target_data[c, :] = pred
            else:
                X_data = data[c, :]
                poly_features = PolynomialFeatures(degree = 2)
                X_poly = poly_features.fit_transform(time_seq.reshape(-1,1))
                poly_model = LinearRegression() 
                poly_model.fit(X_poly, X_data)
                pred = poly_model.predict(X_poly)
                target_data[c, :] = pred
        
    print("Done!!!")
    return target_data


# ## Main Programes

# In[ ]:


## DATA pre-processing

#import datasetsJ
X_import = np.load('Extracted_timeseries_win3.npy')
print("Dimesion of dataset: ", X_import.shape)
print("'''''Import successfully'''''")

# Characteristic of the data
num_win = X_import.shape[2]
num_cond = int(X_import.shape[0]) 
person_num = int(X_import.shape[1])
smp_freq = 250  # 250 Hz
win_len = int(X_import.shape[3]/smp_freq) # 3s  Using sliding window size 3 secs

y_extractedFFT = np.load("PSD_amplitudes_win3.npy")
y_swapped = np.swapaxes(y_extractedFFT, 0, 1)
print(y_swapped.shape)

## Swapping dimension data to easily understand

X_swapped = np.swapaxes(X_import, 0, 1)
print("Dimension of data after applying swappaxes function", X_swapped.shape)

## Leave one out for testing and applying cross-validation training

#Manipulate training data set 
train_people = list(range(person_num))

# Leave one out for Testing
#Choose test person 
# all_person_num.remove(test_person_num)
for test_person_num in train_people:
    if test_person_num >= 10:
        rest_of_training = [p for p in train_people if p != test_person_num]
        # Randomly leave ont for validation set
        for idx_run, vp in enumerate(rest_of_training):
            val_person_num = vp
            remain_person = [k for k in rest_of_training if k!= val_person_num]

            print("==== the test person number #", test_person_num, "====")
            print("==== the validated person number #", val_person_num, "====")

            #Check the remaining which use for training set
            num_train_people = int(len(remain_person))
            print("==== the set of subject training data #", remain_person, "====", "\n")
            
            #Slicing to get training data
            X_train = np.zeros((person_num-2, num_cond, X_swapped.shape[2], X_swapped.shape[3]))
            y_train = np.zeros((person_num-2, num_cond, y_swapped.shape[2]))

            for x, value in enumerate(remain_person):
                X_train[x,:,:,:] = X_swapped[value, :, :, :]
                y_train[x,:,:] = y_swapped[value, :, :]

            #Set up testing set as follwing (test_person_number) 
            X_test = np.zeros((num_cond, X_swapped.shape[2], X_swapped.shape[3]))
            X_test[:,:,:] = X_swapped[test_person_num,:,:,:]

            #Set up validation set as follwing (val_person_number) 
            X_val = np.zeros((num_cond, X_swapped.shape[2], X_swapped.shape[3]))
            X_val[:,:,:] = X_swapped[val_person_num,:,:,:]

            y_val = np.zeros((num_cond, y_swapped.shape[2]))
            y_test = np.zeros((num_cond, y_swapped.shape[2]))

            y_val[:,:] = y_swapped[val_person_num, :, :]
            y_test[:,:] = y_swapped[test_person_num, :, :]

            print("==== the Dimension of training set ====")
            print("Dimension of training data #", X_train.shape, "\n")

            print("==== the Dimension of testing set ====")
            print("Dimension of testing data #", X_test.shape, "\n")

            print("==== the Dimension of validation data ====")
            print("Dimension of validation data #", X_val.shape, "\n")

            print("==== the Dimension of training set for constructing target signals ====")
            print("Dimension of training data #", y_train.shape, "\n")

            print("==== the Dimension of validatio set for constructing target signals ====")
            print("Dimension of training data #", y_val.shape, "\n")

            print("==== the Dimension of testing set for constructing target signals ====")
            print("Dimension of training data #", y_test.shape, "\n")

            ## Min-Max scalar with global max and min from training

            #Calculate global max and global min from training set
            max_values_train = np.zeros((X_train.shape[0], X_train.shape[1]))
            min_values_train = np.zeros((X_train.shape[0], X_train.shape[1]))
            for p in range(X_train.shape[0]):
                for c in range(X_train.shape[1]):
                    max_values_train[p, c] = np.max(X_train[p, c, :, :]) 
                    min_values_train[p, c] = np.min(X_train[p, c, :, :])  

            max_values_train = np.swapaxes(max_values_train, 0, 1)
            min_values_train = np.swapaxes(min_values_train, 0, 1)   

            global_max = max_values_train.mean(axis=1)
            global_min = min_values_train.mean(axis=1)

            print("Global max values from training is: ", global_max)
            print("Global min values from training is: ", global_min)

            #Calculate global max and global min from target set
            max_values_tar = np.zeros((y_train.shape[0], y_train.shape[1]))
            min_values_tar = np.zeros((y_train.shape[0], y_train.shape[1]))
            for p_tar in range(y_train.shape[0]):
                for c_tar in range(y_train.shape[1]):
                    max_values_tar[p_tar, c_tar] = np.max(y_train[p_tar, c_tar, :]) 
                    min_values_tar[p_tar, c_tar] = np.min(y_train[p_tar, c_tar, :])  

            max_values_tar = np.swapaxes(max_values_tar, 0, 1)
            min_values_tar = np.swapaxes(min_values_tar, 0, 1)   

            global_max_tar = max_values_tar.mean(axis=1)
            global_min_tar = min_values_tar.mean(axis=1)

            print("Global max values from training is: ", global_max_tar)
            print("Global min values from training is: ", global_min_tar)

            ## Perform Min-Max scalar on training set

            X_train_normalized = np.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]))
            #Define new scales
            new_max = 1
            new_min = 0
            for per_idx in range(X_train.shape[0]):
                for con_idx in range(X_train.shape[1]):
                    for win_idx in range(X_train.shape[2]):
                        X_std = (X_train[per_idx, con_idx, win_idx, :]-global_min[con_idx])/ (global_max[con_idx] - global_min[con_idx])
                        X_train_normalized[per_idx, con_idx, win_idx, :] = X_std*(new_max-new_min) + new_min
            print(X_train_normalized.shape)

            y_train_normalized = np.zeros((y_train.shape[0], y_train.shape[1], y_train.shape[2]))

            #Define new scales
            new_max = 1
            new_min = 0
            for per_idx_tar in range(y_train.shape[0]):
                for con_idx_tar in range(y_train.shape[1]):
                    y_std = (y_train[per_idx_tar, con_idx_tar, :]-global_min_tar[con_idx_tar])/ (global_max_tar[con_idx_tar] - global_min_tar[con_idx_tar])
                    y_train_normalized[per_idx_tar, con_idx_tar, :] = y_std*(new_max-new_min) + new_min
            print(y_train_normalized.shape)

            ## Perform Min-Max scalar on both testing set and validation set
            X_val_normalized = np.zeros((X_val.shape[0], X_val.shape[1], X_val.shape[2]))
            X_test_normalized = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

            for c_idx in range(X_val.shape[0]):
                for w_idx in range(X_val.shape[1]):
                    X_val_std = (X_val[c_idx, w_idx, :]-global_min[c_idx])/ (global_max[c_idx] - global_min[c_idx])
                    X_test_std = (X_test[c_idx, w_idx, :]-global_min[c_idx])/ (global_max[c_idx] - global_min[c_idx])
                    X_val_normalized[c_idx, w_idx, :] = X_val_std*(new_max-new_min) + new_min
                    X_test_normalized[c_idx, w_idx, :] = X_test_std*(new_max-new_min) + new_min
            print(X_val_normalized.shape)
            print(X_test_normalized.shape)

            y_test_normalized = np.zeros((y_test.shape[0], y_test.shape[1]))
            y_val_normalized = np.zeros((y_val.shape[0], y_val.shape[1]))
            for c_idx_tar in range(y_test.shape[0]):
                for w_idx_tar in range(y_test.shape[1]):
                    y_test_std = (y_test[c_idx_tar, w_idx_tar]-global_min_tar[c_idx_tar])/ (global_max_tar[c_idx_tar] - global_min_tar[c_idx_tar])
                    y_test_normalized[c_idx_tar, w_idx_tar] = y_test_std*(new_max-new_min) + new_min

                    y_val_std = (y_val[c_idx_tar, w_idx_tar]-global_min_tar[c_idx_tar])/ (global_max_tar[c_idx_tar] - global_min_tar[c_idx_tar])
                    y_val_normalized[c_idx_tar, w_idx_tar] = y_val_std*(new_max-new_min) + new_min
            print(y_test_normalized.shape)
            print(y_val_normalized.shape)
            ## Polynomial Regression for curve fitting

            ## Curve fitting function to construct target signals
            target_data = curve_fitting(y_train_normalized)
            target_data_val = curve_fitting(y_val_normalized)
            target_data_test = curve_fitting(y_test_normalized)
            print(target_data.shape)
            print(target_data_val.shape)
            print(target_data_test.shape)

            ## Random Forest Regressor
            ## Data preparation
            X_train_used = X_train_normalized.reshape(-1, X_train_normalized.shape[3])
            X_val_used = X_val_normalized.reshape(-1, X_test_normalized.shape[2])
            X_test_used = X_test_normalized.reshape(-1, X_test_normalized.shape[2])

            #Target signals
            y_train_used = target_data .reshape(-1)
            y_val_used = target_data_val.reshape(-1)
            y_test_used = target_data_test.reshape(-1)

            print("Checking data dimension before performimng training")
            print(X_train_used.shape)
            print(X_val_used.shape)
            print(X_test_used.shape,"\n")

            print("Checking target data dimesion before performimng training")
            print(y_train_used.shape)
            print(y_val_used.shape)
            print(y_test_used.shape,"\n")

            list_C = [0.001, 0.01, 0.1, 1, 10, 100]
            list_ep = [0.001, 0.01, 0.1, 1, 10]
            list_degree = [0,1,2,3,4,5,6]
            
            min_mse = 1
            for deg in list_degree:
                for c_val in list_C:
                    for ep_val in list_ep:
                        model = SVR(kernel='poly', degree=deg, gamma = 'auto', C = c_val, epsilon = ep_val)
                        model.fit(X_train_used, y_train_used)
                        model_predict = model.predict(X_val_used)

                        all_mse = mean_squared_error(y_val_used, model_predict)
                        all_r2_score = my_r2_score(y_val_used, model_predict)
                        print("Poly_degree: {} C_value: {} Epsilon_value:{} All_MSE: {} All_R2: {}".format(deg, c_val, ep_val, all_mse, all_r2_score))

                        if all_mse < min_mse:
                            min_mse = all_mse
                            best_deg = deg
                            best_C = c_val 
                            best_ep = ep_val
                            print("The best poly_degree: {} The best C value is: {} The best epsilon_value:{} The best mse: {}".format(best_deg, best_C, best_ep, all_mse))

            regression = SVR(kernel='poly', degree = best_deg, gamma = 'auto', C=best_C, epsilon = best_ep)
            regression.fit(X_train_used, y_train_used )
            #Monitor time during prediction process
            start = time.time()
            Y_pred = regression.predict(X_test_used)
            end = time.time()
            Monitor_timing = end - start
            print("This model have just taken time into prediction is : ", Monitor_timing)
            
            mse = mean_squared_error(y_test_used, Y_pred)
            rmse = sqrt(mean_squared_error(y_test_used, Y_pred))
            mae = mean_absolute_error(y_test_used, Y_pred)
            r2_score = my_r2_score(y_test_used, Y_pred)
            print("MSE_score: {} RMSE_score:{} All_MAE: {} All_R2: {}".format(mse, rmse, mae, r2_score))
            
    #         plt.title('condition1')
    #         plt.plot(y_test_used[:int(len(y_test_used)/num_cond)], c='b')
    #         plt.plot(Y_pred[:int(len(Y_pred)/num_cond)], 'r')
    #         plt.show()
            mae1 = mean_absolute_error(y_test_used[:int(len(y_test_used)/num_cond)], Y_pred[:int(len(Y_pred)/num_cond)])
            rmse1 = sqrt(mean_squared_error(y_test_used[:int(len(y_test_used)/num_cond)], Y_pred[:int(len(Y_pred)/num_cond)]))
            mse1 = mean_squared_error(y_test_used[:int(len(y_test_used)/num_cond)], Y_pred[:int(len(Y_pred)/num_cond)])
            r2_score1 = my_r2_score(y_test_used[:int(len(y_test_used)/num_cond)], Y_pred[:int(len(Y_pred)/num_cond)])
            print('MSE condition1:', mse1)
            print('R2 condition1:', r2_score1)

    #         plt.title('condition2')
    #         plt.plot(y_test_used[int(len(y_test_used)/num_cond):int(len(y_test_used)/num_cond*2)], c='b')
    #         plt.plot(Y_pred[int(len(Y_pred)/num_cond):int(len(Y_pred)/num_cond*2)],'r')
    #         plt.show()
            mae2 = mean_absolute_error(y_test_used[int(len(y_test_used)/num_cond):int(len(y_test_used)/num_cond*2)], Y_pred[int(len(Y_pred)/num_cond):int(len(Y_pred)/num_cond*2)])
            rmse2 = sqrt(mean_squared_error(y_test_used[int(len(y_test_used)/num_cond):int(len(y_test_used)/num_cond*2)], Y_pred[int(len(Y_pred)/num_cond):int(len(Y_pred)/num_cond*2)]))
            mse2 = mean_squared_error(y_test_used[int(len(y_test_used)/num_cond):int(len(y_test_used)/num_cond*2)], Y_pred[int(len(Y_pred)/num_cond):int(len(Y_pred)/num_cond*2)])
            r2_score2 = my_r2_score(y_test_used[int(len(y_test_used)/num_cond):int(len(y_test_used)/num_cond*2)], Y_pred[int(len(Y_pred)/num_cond):int(len(Y_pred)/num_cond*2)])
            print('MSE condition2:', mse2)
            print('R2 condition2:', r2_score2)

    #         plt.title('condition3')

    #         plt.plot(y_test_used[int(len(y_test_used)/num_cond*2):int(len(y_test_used)/num_cond*3)], c='b')
    #         plt.plot(Y_pred[int(len(Y_pred)/num_cond*2):int(len(Y_pred)/num_cond*3)],'r')
    #         plt.show()
            mae3 = mean_absolute_error(y_test_used[int(len(y_test_used)/num_cond*2):int(len(y_test_used)/num_cond*3)], Y_pred[int(len(Y_pred)/num_cond*2):int(len(Y_pred)/num_cond*3)])
            rmse3 = sqrt(mean_squared_error(y_test_used[int(len(y_test_used)/num_cond*2):int(len(y_test_used)/num_cond*3)], Y_pred[int(len(Y_pred)/num_cond*2):int(len(Y_pred)/num_cond*3)]))
            mse3 = mean_squared_error(y_test_used[int(len(y_test_used)/num_cond*2):int(len(y_test_used)/num_cond*3)], Y_pred[int(len(Y_pred)/num_cond*2):int(len(Y_pred)/num_cond*3)])
            r2_score3 = my_r2_score(y_test_used[int(len(y_test_used)/num_cond*2):int(len(y_test_used)/num_cond*3)], Y_pred[int(len(Y_pred)/num_cond*2):int(len(Y_pred)/num_cond*3)])
            print('MSE condition3:', mse3)
            print('R2 condition3:', r2_score3)

    #         plt.title('condition4')

    #         plt.plot(y_test_used[int(len(y_test_used)/num_cond*3):], c='b')
    #         plt.plot(Y_pred[int(len(Y_pred)/num_cond*3):],'r')
    #         plt.show()
            mae4 = mean_absolute_error(y_test_used[int(len(y_test_used)/num_cond*3):], Y_pred[int(len(Y_pred)/num_cond*3):])
            rmse4 = sqrt(mean_squared_error(y_test_used[int(len(y_test_used)/num_cond*3):], Y_pred[int(len(Y_pred)/num_cond*3):]))
            mse4 = mean_squared_error(y_test_used[int(len(y_test_used)/num_cond*3):], Y_pred[int(len(Y_pred)/num_cond*3):])
            r2_score4 = my_r2_score(y_test_used[int(len(y_test_used)/num_cond*3):], Y_pred[int(len(Y_pred)/num_cond*3):])
            print('MSE condition4:', mse4)
            print('R2 condition4:', r2_score4)
            
            # list_row = range(0,200,10)
            # # Initialize a workbook
            # read_book = open_workbook("Poly_SVR_allcon_win3_revised.xls")
            # write_book = copy(read_book)

            # # Add a sheet to the workbook
            # row = write_book.get_sheet(0)

            # #The fill data on table
            # row.write(idx_run+list_row[test_person_num]+2,0, test_person_num+1)
            # row.write(idx_run+list_row[test_person_num]+2,1, val_person_num+1)
            # row.write(idx_run+list_row[test_person_num]+2,2, best_C)
            # row.write(idx_run+list_row[test_person_num]+2,3, 'None')
            # row.write(idx_run+list_row[test_person_num]+2,4, best_ep)
            # row.write(idx_run+list_row[test_person_num]+2,5, best_deg)
            # row.write(idx_run+list_row[test_person_num]+2,6, Monitor_timing)
            # row.write(idx_run+list_row[test_person_num]+2,7, mse1)
            # row.write(idx_run+list_row[test_person_num]+2,8, rmse1)
            # row.write(idx_run+list_row[test_person_num]+2,9, mae1)
            # row.write(idx_run+list_row[test_person_num]+2,10, r2_score1)
            # row.write(idx_run+list_row[test_person_num]+2,11, mse2)
            # row.write(idx_run+list_row[test_person_num]+2,12, rmse2)
            # row.write(idx_run+list_row[test_person_num]+2,13, mae2)
            # row.write(idx_run+list_row[test_person_num]+2,14, r2_score2)
            # row.write(idx_run+list_row[test_person_num]+2,15, mse3)
            # row.write(idx_run+list_row[test_person_num]+2,16, rmse3)
            # row.write(idx_run+list_row[test_person_num]+2,17, mae3)
            # row.write(idx_run+list_row[test_person_num]+2,18, r2_score3)
            # row.write(idx_run+list_row[test_person_num]+2,19, mse4)
            # row.write(idx_run+list_row[test_person_num]+2,20, rmse4)
            # row.write(idx_run+list_row[test_person_num]+2,21, mae4)
            # row.write(idx_run+list_row[test_person_num]+2,22, r2_score4)
            # write_book.save("Poly_SVR_allcon_win3_revised.xls")


# In[ ]:


print("Done!!!")

