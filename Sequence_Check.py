# 对进行灰色预测的序列数据进行事前与事后检验
import numpy as np


# 事前检验{准光滑检验、级比检验}
def Before_Evaluation(data, evaluation_index="level_check", remove_set=False):
    if evaluation_index == "level_check":
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
    elif evaluation_index == "smooth_check":
        AGO = data.cumsum()
        if sum(data < 0) != 0:
            print("准指数检验序列不可有负数，请对序列做平移变换")
        else:
            rho = [data[i] / AGO[i - 1] for i in range(1, len(data))]
            rhos = [rho[i - 1] / rho[i] for i in range(1, len(rho))]
            per = []  # 符合准指数检验数据的比例
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
            if remove_set:
                if (sum(per[2:]) / (len(per) - 2)) > 0.6:
                    print("通过准指数检验")
                    print('去掉前两个数据后通过准光滑检验的数据占比为：', round(sum(per[2:]) / (len(per) - 2), 2))
                    return True
                else:
                    print("未通过准指数检验")
                    print('去掉前两个数据后通过准光滑检验的数据占比为：', round(sum(per[2:]) / (len(per) - 2), 2))
                    return False
            else:
                if (sum(per) / len(per)) > 0.9:
                    print("通过准指数检验")
                    print('通过准指数检验的数据占比为：', round(sum(per) / len(per), 2))
                    return True
                else:
                    print("未通过准指数检验")
                    print('通过准指数检验的数据占比为：', round(sum(per) / len(per), 2))
                    return False


def translation(data, fuc, n=5, max_iter=500):
    """对不满足事前检验的序列数据做平移变换"""
    pass


# 事后检验, {相对误差、均方差比值、小误差概率、灰色关联度}
def After_Evaluation(true_data, hat_data, evaluation_index="Relative Error"):
    """对灰色预测进行事后检验, {相对误差检验、均方差比检验、小误差概率检验}"""
    e = true_data - hat_data
    re = np.abs(e / true_data)
    re_mean = np.mean(re)
    S1 = np.std(true_data, ddof=1)
    S2 = np.std(e, ddof=1)
    C0 = S2 / S1
    Pe = np.mean(e)
    p_error = np.sum(np.abs(e - Pe) < (0.6745 * S1)) / len(e)
    if evaluation_index == "Relative Error":
        if re_mean <= 0.01:
            print("由相对误差判断模型预测精度为一级，相对误差为", re_mean)
        elif 0.01 < re_mean <= 0.05:
            print("由相对误差判断模型预测精度为二级，相对误差为", re_mean)
        elif 0.05 < re_mean <= 0.10:
            print("由相对误差判断模型预测精度为三级，相对误差为", re_mean)
        elif 0.10 < re_mean <= 0.20:
            print("由相对误差判断模型预测精度为四级，相对误差为", re_mean)
        else:
            print("不建议使用灰色预测模型，相对误差极大")
        return re_mean
    elif evaluation_index == "Mean Variance Ratio":
        if 0 <= C0 < 0.35:
            print("由均方差比值判断模型预测精度为一级，均方差比值为", C0)
        elif 0.35 <= C0 < 0.5:
            print("由均方差比值判断模型预测精度为二级，均方差比值为", C0)
        elif 0.5 <= C0 < 0.65:
            print("由均方差比值判断模型预测精度为三级，均方差比值为", C0)
        elif 0.65 <= C0 <= 0.8:
            print("由均方差比值判断模型预测精度为四级，均方差比值为", C0)
        else:
            print("不建议使用灰色预测模型，均方差比值极大，均方差比值为", C0)
        return C0
    elif evaluation_index == "Probability of Small Error":
        if 0.95 < p_error <= 1:
            print("由小误差概率判断模型预测精度为一级，小误差概率为", p_error)
        elif 0.8 < p_error <= 0.95:
            print("由小误差概率判断模型预测精度为二级，小误差概率为", p_error)
        elif 0.70 < p_error <= 0.8:
            print("由小误差概率判断模型预测精度为三级，小误差概率为", p_error)
        elif 0.6 < p_error <= 0.7:
            print("由小误差概率判断模型预测精度为四级，小误差概率为", p_error)
        else:
            print("不建议使用灰色预测模型，小误差概率极小，小误差概率为", p_error)
        return p_error