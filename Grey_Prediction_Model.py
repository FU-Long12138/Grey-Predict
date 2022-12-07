import numpy as np


class GM11:
    def __init__(self, data):
        self.data = data
        self.AGO = None
        self.Z = None
        self.a = None
        self.b = None
        self.ori_hat = None

    def model_fit(self):
        self.AGO = self.data.cumsum()
        Z = []
        for i in range(1, len(self.AGO)):
            Z.append(0.5 * (self.AGO[i - 1] + self.AGO[i]))
        self.Z = np.array(Z)
        B = np.mat(np.vstack((-self.Z, np.ones(len(self.Z))))).T
        Y = np.mat(np.delete(self.data, 0)).T
        a_hat = np.linalg.inv(B.T * B) * B.T * Y
        a_hat = np.array(a_hat)
        self.a, self.b = float(a_hat[0]), float(a_hat[1])

    def predict(self, pre_times):
        Times = len(self.data) + pre_times

        def pref(t, a=self.a, b=self.b, x0=self.data[0]):
            return (x0 - b / a) * np.exp(-a * (t - 1)) + b / a

        x1_hat = pref(np.arange(1, Times + 1))
        x0_hat = np.hstack((x1_hat[0], np.diff(x1_hat)))
        x0_pre = x0_hat[len(self.data):]
        ori_hat = x0_hat[:len(self.data)]
        self.ori_hat = ori_hat
        return x0_pre

    def evaluation(self, evaluation_index="Relative Error"):
        e = self.data - self.ori_hat
        re = np.abs(e / self.data)
        re_mean = np.mean(re)
        S1 = np.std(self.data, ddof=1)
        S2 = np.std(e, ddof=1)
        C0 = S2 / S1
        Pe = np.mean(e)
        p_error = sum(np.abs(self.data - self.ori_hat - Pe) < (0.6745 * S1)) / len(self.data)
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
            if C0 <= 0.35:
                print("由均方差比值判断模型预测精度为一级，均方差比值为", C0)
            elif 0.35 < C0 <= 0.5:
                print("由均方差比值判断模型预测精度为二级，均方差比值为", C0)
            elif 0.5 < C0 <= 0.65:
                print("由均方差比值判断模型预测精度为三级，均方差比值为", C0)
            elif 0.65 < C0 <= 0.8:
                print("由均方差比值判断模型预测精度为四级，均方差比值为", C0)
            else:
                print("不建议使用灰色预测模型，均方差比值极大，均方差比值为", C0)
            return C0
        elif evaluation_index == "Probability of Small Error":
            if p_error >= 0.95:
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


class Verhulst:
    pass
