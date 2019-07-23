# complete logistic regression
import random
import numpy as np
import matplotlib.pyplot as plt
def sigmod(z):
    y = 1 / (1 + np.exp(z))
    return y
    
def inference(w, b, x):
    z = np.dot(w, x) + b
    return sigmod(z)

def eval_loss(w, b, x_list, y_label):
    _sum_loss = list(map(lambda x, y : y * np.log(inference(w, b, x))
                     + (1 - y) * np.log(1 - inference(w, b, x)),
                     x_list, y_label))
    loss = sum(_sum_loss) / len(x_list)
    return loss

def gradient(pred_y, gt_y, x):
    diff = gt_y - pred_y
    dw = diff * x
    db = diff
    return dw, db

def calc_step_greadient(batch_x_list, gt_y_list, w, b, lr):
    batch_size = len(batch_x_list)
    sum_dw = np.zeros(w.shape)
    sum_db = 0

    for i in range(batch_size):
        pred_y = inference(w, b, batch_x_list[i])
        dw,db = gradient(pred_y, gt_y_list[i], batch_x_list[i])
        sum_dw += dw
        sum_db += db
    
    dw = sum_dw / batch_size
    db = sum_db / batch_size
    w += lr * dw
    b += lr * db
    return w,b

def train(sample_set, label_set, dim, batch_size, lr, max_iter):
    w = np.zeros(dim)
    b = 0
    nlen = sample_set.shape[0]
    for i in range(max_iter):
        batch_ind = np.random.choice(nlen, batch_size)
        x_batch = [sample_set[j, :] for j in batch_ind]
        gt_y = [label_set[j] for j in batch_ind]
        w,b = calc_step_greadient(x_batch, gt_y, w, b, lr)
        if i % 100 == 0:
            print("w = {0} b = {1}".format(w, b))
            print("loss is {0}".format(eval_loss(w, b, sample_set, label_set)))
    return w,b
            
def gen_bmi_data(len):
    body_data = np.zeros([len, 3])
    positive = []
    negtive = []
    for i in range(len):
        body_data[i][0] = random.uniform(1.5, 2.4)
        body_data[i][1] = random.uniform(40, 120)
        if body_data[i][1] / (body_data[i][0]**2) < 22:
            body_data[i][2] = 1 
            positive.append([body_data[i][0], body_data[i][1]])
        else:
            negtive.append([body_data[i][0], body_data[i][1]])

    #plt.figure(1)
    #plt.scatter([i[0] for i in positive], [i[1] for i in positive], marker = "*")
    #plt.scatter([i[0] for i in negtive], [i[1] for i in negtive], marker = "x")
    #plt.show()
    return body_data

if __name__ == "__main__":
    raw_data = gen_bmi_data(500)
    dim = 2
    sample_set = raw_data[:250, 0:2]
    label_set = raw_data[:250, 2]
    batch_size = 50
    lr = 0.001
    max_iter = 10000
    w,b = train(sample_set[:250], label_set[:250], dim, batch_size, lr, max_iter)




