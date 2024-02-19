import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, accuracy_score
import pickle
import config
class ESN(object):
    def __init__(self, inSize=1,resSize=50, rho=0.9, cr=0.05, leaking_rate=0.1, W=None):
        self.inSize=inSize
        self.resSize = resSize
        self.leaking_rate = leaking_rate
        self.s = np.zeros(self.resSize)
        self.Win= np.random.rand(self.resSize, self.inSize) * 2 - 1 
        if W is None:
            N = resSize * resSize
            W = np.random.rand(N) *2-1
            zero_index = np.random.permutation(N)[int(N * cr * 1.0):]
            W[zero_index] = 0
            W = W.reshape((self.resSize, self.resSize))
            print
            'ESN init: Setting spectral radius...',
            rhoW = max(abs(linalg.eig(W)[0]))
            print
            'done.'
            W *= rho / rhoW
        else:
            assert W.shape[0] == W.shape[1] == resSize, "reservoir size mismatch"
        self.W = W

    def get_state(self, X):
        X = X.T
        s = self.s.copy()
        for t, u in enumerate(X):
            s = (1 - self.leaking_rate) * s + self.leaking_rate * \
                np.tanh(np.dot(self.Win, u) + np.dot(self.W, s))
        return s

    def collect_states(self,X):
        S = np.zeros((len(X), self.resSize))
        for idx,x in enumerate(X):
            s=self.get_state(x)
            S[idx]=s
        return S.T  


    def train(self, X, Y, lmbd=1e-6):
        assert len(X) == Y.shape[1], "input lengths mismatch."
        S=self.collect_states(X)
        pseudoInverse_S= np.dot(S.T, np.linalg.inv( np.eye(S.shape[0])/lmbd + np.dot(S, S.T)))
        Wout = np.dot(Y, pseudoInverse_S)
        self.Wout=Wout

    def predict(self, X):
        S = self.collect_states(X)
        Y = np.dot(self.Wout,S)
        return Y

if(__name__=="__main__"):
    run_times=config.run_times
    max_times=config.max_times
    root='D:/Code/stein_ESN'
    esn_array=[]
    for i in range(run_times):
        np.random.seed(i)
        esn=ESN(inSize=13)
        esn_array.append(esn)
    with open('pkl/esn.pkl', 'wb') as f:
        pickle.dump(esn_array,f)


