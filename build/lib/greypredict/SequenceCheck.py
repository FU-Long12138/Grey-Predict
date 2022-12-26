# 对进行灰色预测的序列数据进行事前与事后检验
import numpy as np


# 事前检验{准光滑检验、级比检验}
def Before_Evaluation(data: np.ndarray or list, evaluation_index="level_check", remove_set=False):
    """
    对即将进行灰色预测的数据进行事前检验
    :param data: 即将进行灰色预测的数据
    :param evaluation_index: 评估指标, 可选择{"level_check", "smooth_check"}, 分别为级比检验与准光滑检验
    :param remove_set: 在进行准光滑检验时是否去除前两个数据
    :return: True代表通过检验, False代表未通过检验
    """
    if isinstance(data, list):
        data = np.array(data)
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
                print('级比检验不通过')
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


# 事后检验, {相对误差、均方差比值、小误差概率、灰色关联度}
def After_Evaluation(true_data: np.ndarray or list, hat_data: np.ndarray or list, evaluation_index="Relative Error"):
    """
    对灰色预测进行事后检验
    :param true_data: 真实值
    :param hat_data: 灰色预测的拟合值或预测值
    :param evaluation_index: 评估指标, 提供了{相对误差检验、均方差比检验、小误差概率检验}三种指标,
    对应的参数为{"Relative Error", "Mean Variance Ratio", "Probability of Small Error"}
    :return:
    """
    if isinstance(true_data, list):
        true_data = np.array(true_data)
    if isinstance(hat_data, list):
        hat_data = np.array(hat_data)
    e = true_data - hat_data
    re = np.abs(e / true_data)
    re_mean = np.mean(re)
    S1 = np.std(true_data, ddof=1)
    S2 = np.std(e, ddof=1)
    C0 = S2 / S1
    Pe = np.mean(e)
    p_error = np.sum(np.abs(e - Pe) < (0.6745 * S1)) / len(e)
    if evaluation_index == "Relative Error":
        return re_mean
    elif evaluation_index == "Mean Variance Ratio":
        return C0
    elif evaluation_index == "Probability of Small Error":
        return p_error
