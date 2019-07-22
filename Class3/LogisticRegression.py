# complete logistic regression
import math
import random
def inference(theta, x):
    y = 1 / (1 + math.exp(- theta * x)))
    return y

def eval_loss(theta, x_list, y_label):
    _sum_loss = list(map(lambda x, y : y * math.log(inference(theta, x))
                     + (1 - y) * math.log(1 - inference(theta, x)),
                     x_list, y_label))
    loss = sum(_sum_loss) / len(x_list)
    return loss

def gradient(pred_y, gt_y, x):
    diff = gt_y - pred_y
    diff_theta = diff * x

def calc_step_greadient(batch_x_list, gt_y_list, theta, lr):
    batch_size = len(batch_x_list)
    d_theta = 0

    for i in range(batch_size):
        pred_y = inference(theta, batch_x_list[i])
        d_theta += gradient(pred_y, gt_y_list[i], batch_x_list[i])
    
    d_theta /= batch_size
    theta += lr * d_theta
    return theta

def train(sample_set, label_set, batch_size, lr, max_iter):
    theta = 0
    for i in range(max_iter):
        batch_ind = random.choice(len(sample_set), batch_size)
        x_batch = [sample_set(j) for j in batch_ind]
        gt_y = [label_set[j] for j in batch_ind]
        theta = calc_step_greadient(x_batch, gt_y, theta, lr)
        if i % 100 == 0:
            print("theta{0}".format(theta))
            printf("loss is {0}".format(eval_loss(theta, sample_set, label_set)))
            
def gen_sample_data():
    w = random.randint(0, 10) + random.random()		# for noise random.random[0, 1)
    b = random.randint(0, 5) + random.random()
    num_samples = 100
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = random.randint(0, 100) * random.random()
        y = w * x + b + random.random() * random.randint(-1, 1)
        x_list.append(x)
        y_list.append(y)
    plt.figure(1)
    plt.scatter(x_list, y_list)
    plt.show()
    return x_list, y_list, w, b



