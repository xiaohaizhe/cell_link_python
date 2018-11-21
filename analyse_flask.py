# coding: utf-8

# 智能分析

# 初始化工作
from flask import request, Flask
import json
import os
import sys
import numpy as np


# 全局
app = Flask(__name__)
ANALYSE_TYPE = ["correlation_analyze", "linear_regression"]         # 支持的分析方式
ANALYSE_ITER = 5000                                                 # 迭代次数
ANALYSE_ALPHA = 0.0001                                              # 学习率


@app.route('/type')
def show_type():
    t = {}
    t['support_analyse_type'] = ANALYSE_TYPE
    # print(t)
    return json.dumps(t)


'''
相关性分析接口
计算指标间的相关性系数
当前支持最大3个指标
'''
@app.route('/correlation_analyse', methods=['GET'])
def correlation_analyze():
    req_data = {}
    param = []
    response = {}

    # 解析数据
    req_data = json.loads(request.get_data(as_text=True))
    try:
        param = req_data['params']
    except:
        # 无参数
        response['code'] = 1
        response['msg'] = 'param not found!'
        response['result'] = ''
        return json.dumps(response)

    # 检查数据长度
    for i in range(1, len(param)):
        if len(param[i-1]) != len(param[i]):
            response['code'] = 1
            response['msg'] = 'number of params is not equal.'
            response['result'] = ''
            return json.dumps(response)

    # 分析
    # np.cov(param)
    result = np.corrcoef(param)

    # 返回
    response['code'] = 0
    response['msg'] = 'correlation analyse success!'
    response['result'] = result.tolist()
    return json.dumps(response)


# 线性回归
def liner_Regression(data_x, data_y, learningRate=ANALYSE_ALPHA, Loopnum=ANALYSE_ITER):
    weight = np.ones(shape=(1, data_x.shape[1]))
    baise = np.array([[1]])

    for num in range(Loopnum):
        WXPlusB = np.dot(data_x, weight.T) + baise

        loss = np.dot((data_y-WXPlusB).T, data_y-WXPlusB)/data_y.shape[0]
        w_gradient = -(2/data_x.shape[0])*np.dot((data_y-WXPlusB).T, data_x)
        baise_gradient = -2*np.dot((data_y-WXPlusB).T, np.ones(shape=[data_x.shape[0], 1]))/data_x.shape[0]

        weight = weight-learningRate*w_gradient
        baise = baise-learningRate*baise_gradient

        # if num%50==0:
        #    print(loss)       #每迭代50次输出一次loss
    return (weight, baise)


'''
线性回归分析接口
'''
@app.route('/linear_regression', methods=['GET'])
def linear_regression():
    req_data = {}
    param_out = []
    param_in = []
    response = {}

    # 解析数据
    req_data = json.loads(request.get_data(as_text=True))
    try:
        param_in = req_data['input']
        param_out.append(req_data['output'])

    except:
        # 无参数
        response['code'] = 1
        response['msg'] = 'param not found!'
        response['result'] = ''
        return json.dumps(response)

    # 检查数据长度
    for i in range(len(param_in)):
        if len(param_in[i]) != len(param_out[0]):
            response['code'] = 1
            response['msg'] = 'number of params is not equal.'
            response['result'] = ''
            return json.dumps(response)

    # 分析
        result_w, result_b = liner_Regression(np.array(param_in).T,np.array(param_out).T)
        result = result_b.tolist() + result_w.tolist()

    # 返回
    response = {}
    response['code'] = 0
    response['msg'] = 'linear regression analyse success!'
    response['result'] = result
    return json.dumps(response)


# 启动服务
if __name__ == '__main__':
    app.run()
