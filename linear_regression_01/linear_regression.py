import numpy as np

def computer_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #计算均方差
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #计算b的梯度
        b_gradient += (2 / N) * ((w_current * x + b_current) - y)
        #计算w的梯度
        w_gradient += (2 / N) * x * ((w_current * x + b_current) - y)
    #更新w
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]

def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    #学习率
    learning_rate = 0.0001
    #初始化b的值为0
    initial_b = 0
    #初始化权重值为0
    initial_w = 0
    #迭代次数
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w, computer_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running.....")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".format(num_iterations, b, w, computer_error_for_line_given_points(b, w, points)))
    

if __name__ == '__main__':
    run()
    