import numpy as np


def smooth_check(data):
    """准光滑检验"""
    AGO = data.cumsum()
    if sum(data < 0) != 0:
        print("准光滑检验序列不可有负数，请对序列做平移变换")
    else:
        rho = [data[i] / AGO[i - 1] for i in range(1, len(data))]
        rhos = [rho[i - 1] / rho[i] for i in range(1, len(rho))]
        per = []  # 符合准光滑检验数据的比例
        pers = []
        for i in rho:
            if 0 < i < 0.5:
                per.append(1)
            else:
                per.append(0)
        for i in rhos:
            if i < 1:
                pers.append(1)
            else:
                pers.append(0)
        if sum(per) == len(per) and sum(pers) == len(pers):
            print('通过准光滑检验')
            return True
        else:
            print('未通过准光滑检验')
            print('通过准光滑检验的数据占比为：', round(sum(per) / len(per), 2))
            return False


def level_check(data):
    """级比检验"""
    if sum(data < 0) != 0:
        print('级比检验序列不可有负数，请对序列数据做平移变换')
    else:
        n = len(data)
        lambdas = []
        trues = []
        for i in range(1, n):
            lambdai = data[i - 1] / data[i]
            if np.exp(-2 / (n + 1)) < lambdai < np.exp(2 / (n + 1)):
                trues.append(1)
            else:
                trues.append(0)
            lambdas.append(lambdai)
        if 0 in trues:
            print('级比检验不通过，请对序列数据做平移变换')
            return False
        else:
            print('级比检验通过')
            return True


def translation(data, n=5):
    """对不满足级比检验的序列数据做平移变换"""
    pass
