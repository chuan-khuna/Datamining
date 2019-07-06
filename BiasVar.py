import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
import random

class BiasVar:
    def __init__(self, model_weights, x, fx):
        '''
            weights: model weight in list of [w0 w1 w2 ....]
            x: input [x0 x1 x2 ....]
            y: real output
        '''
        self.weights = model_weights
        self.x = x
        self.y = fx
        
        self.cal_mean_weight()
        self.cal_bias()
        self.cal_variance()
    
    def cal_bias(self):
        self.meanweight_res = self.cal_val(self.x, self.mean_weight)
        bias = np.mean(
            (self.meanweight_res - self.y) ** 2
        )
        self.bias = round(bias, 4)
        return self.bias

    def cal_variance(self):
        all_model_res = []
        for w in self.weights:
            all_model_res.append(
                (self.x).dot( np.array(w).T )
            )
        all_model_res = np.array(all_model_res)
        repeat_mean_w_res = np.repeat(np.array([self.meanweight_res]) , repeats=len(self.weights), axis=0)
        variance = np.mean(
            (all_model_res - repeat_mean_w_res)**2
        )
        self.variance = round(variance, 4)
        return self.variance
            

    def cal_mean_weight(self):
        mean_w = []
        for col in range(len(self.weights[0])):
            w = np.mean(self.weights[:, col])
            mean_w.append(w)
        self.mean_weight = np.round(np.array(mean_w), 4)

    def cal_val(self, x, w):
        result = x.dot(w.T)
        return np.round(np.array(result), 4)



if __name__ == "__main__":
    sin_sample = 200
    x = np.linspace(0, 2*np.pi, sin_sample)
    fx = np.sin(x)
    x_aug = np.array([[1, i] for i in x])
    num_model = 20
    const_weights = np.random.uniform(-1, 1, num_model)
    const_weights = np.array([[w, 0] for w in const_weights])
    const_model = BiasVar(const_weights, x_aug, fx)

    print("------ constant model ------")
    print("constant gbar: {}".format(const_model.mean_weight))
    print("constant bias^2: {}".format(const_model.bias))
    print("constant variance: {}".format(const_model.variance))
    print("constant bias^2+variance: {}".format(const_model.bias+const_model.variance))

    # first and second point index for linear model
    fi = random.sample(range(sin_sample), num_model)
    si = random.sample(range(sin_sample), num_model)

    lin_weights = []
    # create linear models, 
    for i in range(num_model):
        x1 = x[fi[i]]
        y1 = fx[fi[i]]
        while fi[i] == si[i]:
            # random agian if same index
            si[i] = random.randint(0, len(fi)-1)
        x2 = x[si[i]]
        y2 = fx[si[i]]
        
        m = (y1 - y2)/(x1 - x2)
        c = y1 - m*x1
        model_w = [c, m]
        lin_weights.append(model_w)
    lin_weights = np.array(lin_weights)
    lin_model = BiasVar(lin_weights, x_aug, fx)
    print("------ linear model ------")
    print("linear gbar: {}".format(lin_model.mean_weight))
    print("linear bias^2: {}".format(lin_model.bias))
    print("linear variance: {}".format(lin_model.variance))
    print("linear bias^2+variance: {}".format(lin_model.bias+lin_model.variance))