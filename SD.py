import numpy as np


def stein_score_function(s,S,h):
    s=s.reshape((-1,1))
    denominators=0
    numrators=0
    for i in range(S.shape[1]):
        s_i=S[:,i].reshape((-1,1))
        denominator=np.exp(np.dot((s-s_i).T,(s-s_i))/(-2*h**2))
        numrator=denominator*(s-s_i)/(-h**2)
        denominators+=denominator
        numrators+=numrator
    return numrators/denominators

def f(s,s_pie):
    return np.dot(s.T,s_pie)

def f_derevative_s(s,s_pie):
    return s_pie


def f_derevative_s_pie(s,s_pie):
    return s

def f_derevative_s_pie_s(s,s_pie):
    return np.eye(s.shape[0])

def get_stein_discrepancy(source_S,target_S,h):
    stein_score_array=[]
    for i in range(target_S.shape[1]):
        s = target_S[:, i].reshape((-1, 1))
        stein_score_array.append(stein_score_function(s,source_S,h))
    u_q_array=np.zeros((target_S.shape[1],target_S.shape[1]))
    f_array = np.zeros((target_S.shape[1], target_S.shape[1]))
    for i in range(target_S.shape[1]):
        s = target_S[:, i].reshape((-1, 1))
        for j in range(i,target_S.shape[1]):
            s_pie=target_S[:,j].reshape((-1,1))
            u_q_array[i][j]=np.dot(stein_score_array[i].T,stein_score_array[j])*f(s,s_pie)+\
                np.dot(stein_score_array[i].T,f_derevative_s_pie(s,s_pie))+\
                np.dot(f_derevative_s(s,s_pie).T,stein_score_array[j])+\
                np.trace(f_derevative_s_pie_s(s,s_pie))
            f_array[i][j]=f(s,s_pie)
    for i in range(target_S.shape[1]):
        for j in range(i):
            u_q_array[i][j]=u_q_array[j][i]
            f_array[i][j]=f_array[j][i]
    numerator=np.sum(u_q_array)
    denominator=np.sum(f_array)
    return numerator/denominator








