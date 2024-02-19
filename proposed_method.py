import dataloader
from model import ESN
import SD
import config

import pickle
import numpy as np
from sklearn.metrics import accuracy_score

import logging
import os
import sys

def one_hot(x, class_count):
    return np.identity(class_count)[x, :]

def organize_data(samples: list):
    X = []
    Y = []
    for sample in samples:
        X.append(sample['X'])
        Y.append(one_hot(sample['y'], class_count))
    Y = np.array(Y).T
    return X, Y

def train(src_train_S, src_train_Y, tgt_train_S, tgt_train_Y, c1, c2):
    beita1 = c1 * np.dot(src_train_Y, src_train_S.T) + c2 * np.dot(tgt_train_Y, tgt_train_S.T)
    beita2 = np.linalg.inv(c1 * np.dot(src_train_S, src_train_S.T) + c2 * np.dot(tgt_train_S, tgt_train_S.T) + np.eye(
        tgt_train_S.shape[0]))
    return np.dot(beita1, beita2)

def test(esn, src_train_X, src_train_Y, tgt_train_X, tgt_train_Y):
    src_train_S = esn.collect_states(src_train_X)
    tgt_train_S = esn.collect_states(tgt_train_X)
    for theta in np.arange(0, 1 + 0.00001, 1 / 100):
        reconstructed_S = []
        j = 0
        for i in range(src_train_S.shape[1]):
            src_s = src_train_S[:, i]
            src_y = src_train_Y[:, i]
            src_y_cls = np.argmax(src_y)
            while (np.argmax(tgt_train_Y[:, j % tgt_train_num]) != src_y_cls):
                j += 1
            tgt_s = tgt_train_S[:, j % tgt_train_num]
            reconstructed_s = (1 - theta) * src_s + theta * tgt_s
            reconstructed_S.append(reconstructed_s)
        reconstructed_S = np.array(reconstructed_S).T
        d= SD.get_stein_discrepancy(reconstructed_S, tgt_train_S, h)
        print(f'theta={theta},d2={d}')

def get_new_source(esn, src_train_X, src_train_Y, tgt_train_X, tgt_train_Y,error):
    src_train_S = esn.collect_states(src_train_X)
    tgt_train_S = esn.collect_states(tgt_train_X)
    D_S0_T = SD.get_stein_discrepancy(src_train_S, tgt_train_S, h)
    D_T_T = SD.get_stein_discrepancy(tgt_train_S, tgt_train_S, h)
    logging.info(f'D_S0_T={D_S0_T},D_T_T={D_T_T}')
    pre_D_S_T = 100000
    while (True):
        D_S_T = SD.get_stein_discrepancy(src_train_S, tgt_train_S, h)
        logging.info(f'stein discrepancy is {D_S_T}')
        if (pre_D_S_T - D_S_T < error ):
            break
        else:
            reconstructed_S = []
            j = 0
            for i in range(src_train_S.shape[1]):
                src_s = src_train_S[:, i]
                src_y = src_train_Y[:, i]
                src_y_cls = np.argmax(src_y)
                while (np.argmax(tgt_train_Y[:, j % tgt_train_num]) != src_y_cls):
                    j += 1
                tgt_s = tgt_train_S[:, j % tgt_train_num]
                theta= np.random.beta(a=beita_param, b=beita_param, size=1)
                theta=max(theta,1-theta)
                reconstructed_s=theta*src_s+(1-theta)*tgt_s
                reconstructed_S.append(reconstructed_s)
            reconstructed_S = np.array(reconstructed_S).T
            src_train_S = reconstructed_S
            pre_D_S_T = D_S_T
    return src_train_S, tgt_train_S


def train(src_train_S, src_train_Y, tgt_train_S, tgt_train_Y, c1, c2):
    beita1 = c1 * np.dot(src_train_Y, src_train_S.T) + c2 * np.dot(tgt_train_Y, tgt_train_S.T)
    beita2 = np.linalg.inv(c1 * np.dot(src_train_S, src_train_S.T) + c2 * np.dot(tgt_train_S, tgt_train_S.T) + np.eye(
        tgt_train_S.shape[0]))
    return np.dot(beita1, beita2)

def get_top_indices(lst,runtimes):
    indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:runtimes]
    return indices

if (__name__ == '__main__'):
    tgt_train_num = config.tgt_train_num
    class_count = config.class_count
    max_times=config.max_times
    run_times = config.run_times
    cs = config.cs
    h = config.h
    log_root=config.log_root

    beita_param=sys.argv[1]
    beita_param=float(beita_param)

    if(not os.path.exists(log_root)):
        os.mkdir(log_root)
    log_dir = f'{log_root}/proposed_method_{beita_param}_{tgt_train_num}.txt'
    logging.basicConfig(level=logging.INFO, filename=log_dir, filemode='w',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logging.info(f'the parameter of beita distribution={beita_param}')
    train_filepath = 'data/Train_Arabic_Digit.txt'
    test_filepath = 'data/Test_Arabic_Digit.txt'
    src_train_and_val_samples, tgt_train_and_val_samples = dataloader.loaddata(train_filepath)
    src_test_samples, tgt_test_samples = dataloader.loaddata(test_filepath)
    src_samples = src_train_and_val_samples + src_test_samples
    tgt_samples = tgt_train_and_val_samples + tgt_test_samples
    with open(f'pkl/esn.pkl', 'rb') as f:
        esn_array = pickle.load(f)
    acc_array=[]
    for i in range(run_times):
        logging.info(f'********************the {i+1}th experiments:******************************')
        np.random.seed(i)
        np.random.shuffle(src_samples)
        np.random.shuffle(tgt_samples)
        src_train_num=int(len(src_samples)*0.8)
        src_val_num=int(len(src_samples)*0.1)
        src_test_num = int(len(src_samples) * 0.1)
        tgt_train_num=tgt_train_num
        tgt_val_num=(len(tgt_samples)-tgt_train_num)//2
        tgt_test_num =(len(tgt_samples)-tgt_train_num)//2
        src_train_samples = src_samples[:src_train_num]
        src_val_samples = src_samples[src_train_num:src_train_num + src_val_num]
        src_test_samples = src_samples[src_train_num + src_val_num:]
        tgt_train_samples = tgt_samples[:tgt_train_num]
        tgt_val_samples = tgt_samples[-(tgt_val_num + tgt_test_num):-tgt_test_num]
        tgt_test_samples = tgt_samples[-tgt_test_num:]
        logging.info(f'source domain: train num={len(src_train_samples)},val_num={len(src_val_samples)},test_num={len(src_test_samples)}')
        logging.info(f'target domain: train num={len(tgt_train_samples)},val_num={len(tgt_val_samples)},test_num={len(tgt_test_samples)}')
        src_train_X, src_train_Y = organize_data(src_train_samples)
        src_val_X, src_val_Y = organize_data(src_val_samples)
        src_test_X, src_test_Y = organize_data(src_test_samples)
        tgt_train_X, tgt_train_Y = organize_data(tgt_train_samples)
        tgt_val_X, tgt_val_Y = organize_data(tgt_val_samples)
        tgt_test_X, tgt_test_Y = organize_data(tgt_test_samples)
        bst_acc, bst_c1, bst_c2,bst_error= 0, 0, 0, 0
        esn = esn_array[i]
        for error in [0.01,0.1,1,10]:
            src_train_S, tgt_train_S = get_new_source(esn, src_train_X, src_train_Y, tgt_train_X, tgt_train_Y,error)
            for idx1, c1 in enumerate(cs):
                c1=10**c1
                for idx2, c2 in enumerate(cs):
                    c2=10**c2
                    Wout = train(src_train_S, src_train_Y, tgt_train_S, tgt_train_Y, c1, c2)
                    tgt_val_S = esn.collect_states(tgt_val_X)
                    predicted_Y = np.dot(Wout, tgt_val_S)
                    tgt_val_Y_cls = [np.argmax(tgt_val_Y[:, i]) for i in range(tgt_val_Y.shape[1])]
                    predicted_Y_cls = [np.argmax(predicted_Y[:, i]) for i in range(predicted_Y.shape[1])]
                    acc = accuracy_score(tgt_val_Y_cls, predicted_Y_cls)
                    if(acc>bst_acc):
                        bst_acc=acc
                        bst_c1=c1
                        bst_c2=c2
                        bst_error=error
        logging.info(f"the biggest acc of validation set:{bst_acc}")
        logging.info(f"the corresponding parameter:{bst_c1,bst_c2,bst_error}")
        src_train_S, tgt_train_S = get_new_source(esn, src_train_X, src_train_Y, tgt_train_X, tgt_train_Y,bst_error)
        Wout = train(src_train_S, src_train_Y, tgt_train_S, tgt_train_Y, bst_c1, bst_c2)
        tgt_test_S = esn.collect_states(tgt_test_X)
        predicted_Y = np.dot(Wout, tgt_test_S)
        tgt_test_Y_cls = [np.argmax(tgt_test_Y[:, i]) for i in range(tgt_test_Y.shape[1])]
        predicted_Y_cls = [np.argmax(predicted_Y[:, i]) for i in range(predicted_Y.shape[1])]
        acc = accuracy_score(tgt_test_Y_cls, predicted_Y_cls)
        logging.info(f"the test acc={acc}")
        acc_array.append(acc)
    acc_array=np.array(acc_array)
    mean=np.mean(acc_array)
    std=np.std(acc_array)
    logging.info(f'test set:mean={mean},std={std}')









