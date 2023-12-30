import argparse
# import imp
import importlib as imp
from operator import imod, index, mod
from re import I
from turtle import forward
import torch
import numpy as np
import os, json, cv2, random
import xml.etree.ElementTree as ET
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import json
import random
import time
from torch.autograd import Variable
# from graphviz import Digraph


from crop_pic_sin import crop_and_filter_objects
# from timm.data import Mixup

from models.rel_models import OnClassify_v1
# from demo import ImageTool
from models.reasoning_out_and_in import Reasoning
from models.reasoning_out_and_in import id2rel
from models.reasoning_out_and_in import ImageTool
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import PIL
from torchviz import make_dot

DATA_INPUT = r"F:\study\3_5programtest\7_bolt_change_v2\data_plastic\org_backdrop_sort\1_plastic_data_set/"  #用于生成训练集与测试集路径

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")
imgtool = ImageTool()


# def make_dot(var, params=None):
#     if params is not None:
#         assert isinstance(params.values()[0], Variable)
#         param_map = {id(v): k for k, v in params.items()}
#
#     node_attr = dict(style='filled',
#                      shape='box',
#                      align='left',
#                      fontsize='12',
#                      ranksep='0.1',
#                      height='0.2')
#     dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
#     seen = set()
#
#     def size_to_str(size):
#         return '(' + (', ').join(['%d' % v for v in size]) + ')'
#
#     def add_nodes(var):
#         if var not in seen:
#             if torch.is_tensor(var):
#                 dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
#             elif hasattr(var, 'variable'):
#                 u = var.variable
#                 name = param_map[id(u)] if params is not None else ''
#                 node_name = '%s\n %s' % (name, size_to_str(u.size()))
#                 dot.node(str(id(var)), node_name, fillcolor='lightblue')
#             else:
#                 dot.node(str(id(var)), str(type(var).__name__))
#             seen.add(var)
#             if hasattr(var, 'next_functions'):
#                 for u in var.next_functions:
#                     if u[0] is not None:
#                         dot.edge(str(id(u[0])), str(id(var)))
#                         add_nodes(u[0])
#             if hasattr(var, 'saved_tensors'):
#                 for t in var.saved_tensors:
#                     dot.edge(str(id(t)), str(id(var)))
#                     add_nodes(t)
#
#     add_nodes(var.grad_fn)
#     return dot


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--finetune', default=r'F:\study\3_5programtest\7_bolt_change_v2\model\mae_pretrain_vit_base.pth',
                        help='finetune from checkpoint') #添加一个参数，用于指定微调模型时的预训练模型的路径。
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    return parser
# --model：模型名称，指定要训练或评估的图像分类模型
# --cutmix_minmax：最小和最大比率，用于覆盖 alpha（混合比例），如果设置则启用 cutmix。
# --finetune：预训练模型的路径，用于微调。
# --drop_path: Drop path 的比例，是一个浮点数，默认为 0.1。Drop path 是一种正则化技术，用于防止过拟合。
# --global_pool: 是否使用全局池化（global pooling）。如果设置，将使用全局平均池化进行分类。默认为 True。
# --cls_token: 是否使用类别令牌（class token）而不是全局池化进行分类。如果设置，将使用类别令牌进行分类。默认为 False。
# --eval: 是否执行仅评估的操作。如果设置，将执行评估操作而不是训练。
# --dist_eval: 是否启用分布式评估。在训练期间，分布式评估可能更快。
# --device: 指定用于训练或测试的设备，默认为 'cuda'。可以设置为 'cpu' 或其他 PyTorch 支持的设备。

def train(model):
    #实现了一个简单的训练和测试过程。训练函数，接受一个模型 model 作为输入。在这个函数中，完成了模型的训练和测试。
    #其中模型使用了交叉熵损失函数、Adam 优化器，并对数据集进行了随机打乱。在每个 epoch 中，都会输出训练和测试的损失
    #以及测试的准确率。如果测试准确率达到某个阈值，则保存当前模型。
    train_set = TripleDataset(DATA_INPUT + 'attribute_black_train.tsv')
    test_set = TripleDataset(DATA_INPUT + 'attribute_black_test.tsv')  # use triple
    #用于训练和测试的数据集。使用 TripleDataset 类加载训练集和测试集。
    lr = 0.001
    epoch_num = 15
    loss_function = nn.MSELoss() # 超参数：损失函数
    # loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #超参数：优化器
    torch.set_grad_enabled(True) #开启梯度计算。
    model.train() #将模型设置为训练模式。

    model.to(device) #将模型移到指定的设备（例如 GPU）上。

    running_loss = 0 #用于累积训练过程中的损失。

    index_list = list(range(train_set.len)) #训练样本的索引列表，通过打乱的方式获取。
    random.shuffle(index_list) #是训练样本的索引列表，通过打乱的方式获取。
    # print(index_list)

    for epoch in range(epoch_num): #迭代 epoch_num 次。外层迭代，表示训练过程的轮次。
        print('------- Epoch', epoch, '-------')
        ans_pos = 0
        ans_neg = 0
        train_loss = 0
        loss_pos = 0
        loss_neg = 0
        acc = 0
        train_pos_cnt = 0
        train_neg_cnt = 0
        test_pos_cnt = 0
        test_neg_cnt = 0
        # 观测度量
        test_loss = 0
        pred_tot = 0
        pred_pos = 0
        pred_neg = 0
        # 性能度量
        acc = 0
        acc_pos = 0
        acc_neg = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        best_score = 0.98
        for i in index_list: #内层迭代，遍历索引列表，获取训练样本。此时i为打乱后第一个样本的原序号
            (img, op_lists, answers,img_file_path) = train_set[i] #获取训练样本的图像、操作列表、答案和图像文件路径。
            for i in range(2): #对于每个样本，执行两次迭代
                optimizer.zero_grad() #清零梯度
                # start_time = time.time()
                # print("start time begin")
                y_pred = model(op_lists[i], img,img_file_path, mode='train') #使用模型进行前向传播
                model.concept_matrix2zero() #模型的 concept_matrix 置零。
                

                loss = loss_function(y_pred, answers[i]) #计算损失并进行反向传播和参数更新。

                loss.requires_grad_(True)
                train_loss += loss.data #累加训练过程中的损失。
                loss.backward() #反向传播。
                optimizer.step() #参数更新。

        # ------------------- test model ---------------
        # test_set = train_set
        for i in range(test_set.len): #执行测试阶段
            (img, op_lists, answers,img_file_path) = test_set[i] #获取测试样本。
            for i in range(2): #对于每个样本，执行两次迭代。
                y_pred = model(op_lists[i], img,img_file_path, mode='train') #对测试集的每个样本，使用模型进行前向传播。
                model.concept_matrix2zero() #将模型的 concept_matrix 置零。
                loss = loss_function(y_pred, answers[i]) #计算测试损失。
                test_loss += loss.data
                y_pred = model(op_lists[i], img,img_file_path,  mode='test') #对测试集的每个样本，使用模型进行前向传播。
                model.concept_matrix2zero()
                # print('预测结果',y_pred) #输出测试集上的预测结果。

                # print('真实结果',answers[i]) #输出测试集上的真实答案。
            # acc compute
                if y_pred.equal(answers[i]) :
                    acc += 1


        # print('[INFO] pos cnt', train_pos_cnt, 'neg cnt', train_neg_cnt)
        # print('[INFO] pos ans', ans_pos, 'neg ans', ans_neg)
        print('[INFO] ---train---')
        print('[INFO]---- train loss ----:', train_loss /( train_set.len*2))
        # print('[INFO]---- train pos loss ----:', loss_pos / train_pos_cnt)
        # print('[INFO]---- train neg loss ----:', loss_neg / train_neg_cnt)
        print('[INFO] ---test---')
        # print('[INFO]---- pred avg ----:', pred_tot / test_set.len)
        # print('[INFO]---- pred avg pos----:', pred_pos / test_pos_cnt)
        # print('[INFO]---- pred avg neg----:', pred_neg / test_neg_cnt)
        print('[INFO]---- test loss ----:', test_loss / (test_set.len*2))
        print('[INFO] ---eval---')
        print('[INFO]---- test acc ----:', acc / (test_set.len*2))
        # print('[INFO]---- test acc pos----:', acc_pos / test_pos_cnt)
        # print('[INFO]---- test acc neg----:', acc_neg / test_neg_cnt)
        # print('[INFO]---- test P ----:', P)
        # print('[INFO]---- test R ----:', R)
        # print('[INFO]---- test F1 ----:', F1)

        # if F1 >= best_score:
        if round(acc / (test_set.len*2), 2) >= best_score:
            #如果测试准确率（acc）达到某个阈值（best_score），则保存当前模型。
            # best_score = round(F1, 2)
            best_score = round(acc / (test_set.len*2), 2)
            name_str = './test_checkpoint_black_epoch15/model_black_best_4neg.pkl'.replace('best', str(best_score))
            torch.save(model, name_str) #如果测试准确率达到某个阈值，则保存当前模型。
            # infer_checkpoint(model)
            # break


def iii():
    #在这个代码中，推理的目标是输出每个属性的推理结果。
    attribute_out_dic={0:'noblack1',1:'black',2:'noblack2',3:'noblack3'} #一个字典，将索引映射到相应的属性名称。例如，0 对应 'hex1'，1 对应 'round1'，以此类推。
    # model = torch.load('./checkpoint/reason_model_zero_1.0_4neg.pkl', map_location=torch.device(device))
    model = torch.load(r"F:\study\3_5programtest\7_bolt_change_v2\test_checkpoint_black_epoch15\model_black_1.0_4neg.pkl",map_location=torch.device(device))
    #从指定路径加载模型。这里加载的模型是训练过程中表现最好的模型，它用于推理。
    for i in range(len(attribute_out_dic)): #遍历属性字典中的每个属性。

        infer_checkpoint(model,attribute_out_dic[i])
        #调用 infer_checkpoint 函数进行推理，传递加载的模型和当前属性名称。


def infer_checkpoint(model,atribute_out): #推理的函数
    #一个字典，将属性名称映射到相应的螺栓类型。例如，'hex1' 对应 'out_hex_bolt'，'round1' 对应 'in_hex_bolt'，以此类推。
    attribute_out2bolt={'noblack1':'pe_notrans_noblack','black':'pp_black_notrans_black','noblack2':'pvc_trans_noblack','noblack3':'abs_notrans_noblack'}
    attribute_out2index={'noblack1':0,'black':1,'noblack2':0,'noblack3':0} #这里对应着哪个位置的标签索引，其中，0位置对应标签0，1位置对应标签1
    # 一个字典，将属性名称映射到相应的索引。例如，'hex1' 对应 0，'round1' 对应 1，以此类推。
    op_list = [
            {'op': 'objects', 'param': ''},
            {'op':'filter_nearest_obj', 'param': ''},
            {'op':'obj_attibute', 'param': [0,attribute_out2index[atribute_out]]}
    ] #一个包含操作的列表，每个操作是一个字典，指定了操作的类型和参数
    img_list = list(range(209, 219)) #一个包含待推理的图像ID的列表。在这里，img_list 中只包含一个图像ID。
    # img_list = [1]
    print('[INFO]---------- 评分测试 ---------')
    tol_num=0
    val_num=0
    for img_id in img_list: #遍历 img_list 中的每个图像ID：
        # if img_id == 43:
        #     continue
        # print('[INFO]---case', img_id, '----')
        img_file_path = DATA_INPUT + attribute_out2bolt[atribute_out]+'/' + str(img_id).zfill(3) + '.jpg'
        #构建图像文件路径 (img_file_path)，用于加载图像。
        # ann_file = DATA_INPUT + 'Annotation/shut-' + str(img_id).zfill(3) + '.xml'
        # print(img_file)
        img = imgtool.load_img(img_file_path)
        #调用 imgtool.load_img 函数加载图像。
        # img_file_path=""
        y_pred = model(op_list, img,img_file_path, mode='test')#输出推理结果 (y_pred)。这里的推理结果是一个包含预测值的张量。
        model.concept_matrix2zero() #将模型的 concept_matrix 置零。
        # max_val,index = torch.max(y_pred,dim=1)
        # print(index)
        tol_num+=1
        print( y_pred[0].data)
        if y_pred[0].data==1:
            val_num+=1
        # print('[INFO] image:', img_id, 'attribute',y_pred)
    print('[INFO] attributu', atribute_out,'total num:', tol_num, 'correct num',val_num, 'accuracy',val_num/tol_num)
    #最终，函数输出属性名称、总数、正确数和准确率。
    # img_file_path ="/home/ur/Desktop/attribute_infer/bolt/data-end2end-triple/true_mul_bolt_crops/in_hex_bolt/003.jpg"

    # img = imgtool.load_img(img_file_path)

    # y_pred = model(op_list, img,img_file_path, mode='test')
    
    # print( y_pred[0].data)

class TripleDataset(Dataset): #用于加载训练和测试数据。
    def __init__(self, file):
        #这是初始化方法，接受一个文件路径 file 作为参数。
        # self.eye_matrix = torch.eye(1)
        self.imgtool = ImageTool() #创建 ImageTool 类的对象 imgtool，用于图像加载和处理。

        # with open(file) as f:
        #     triples = json.load(f)
        # f.close()
        triples_pos = [] #存储正例数据，每个元素是一个元组，包含图像路径和属性信息。
        with open(file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # if i == 0:
                #     continue
                (img_path, attribute_in) = line.split(' ')  # dataset triple 格式
                triples_pos.append((img_path, int(attribute_in)))
                #读取数据文件，将每一行的图像路径和属性信息提取出来，存储在 triples_pos 中。这里的 triples_pos 存储的是正例的数据。

        # # 负采样
        # rate = 2  # 负采样比例
        # triples_neg = []
        # for triple in triples_pos:
        #     (img_id, obj_id, sub_id, rel_id) = triple
        #
        #     # 读取图片中物体数量
        #     img_file = DATA_INPUT + 'images/shut-' + str(img_id).zfill(3) + '.jpg'
        #     ann_file = DATA_INPUT + 'Annotation/shut-' + str(img_id).zfill(3) + '.xml'
        #     # print(img_file)
        #     img, ann = imgtool.load_img(img_file, ann_file)
        #     obj_num = len(ann)
        #
        #     cnt = 0
        #
        #     while (cnt < rate):
        #         # 打乱头尾实体，构造负例
        #         neg_sub_id = random.randint(0, obj_num - 1)
        #         neg_obj_id = random.randint(0, obj_num - 1)
        #         # neg_obj_id = obj_id
        #         neg_triple = (img_id, neg_obj_id, neg_sub_id, rel_id)
        #         if (neg_triple not in triples_pos) and (neg_triple not in triples_neg):
        #             triples_neg.append(neg_triple)
        #             cnt += 1
        #             # print(neg_triple)
        # print('[INFO] neg sample finish')
        # # print(len(triples_pos), len(triples_neg))

        self.triples_pos = triples_pos
        # self.triples_neg = triples_neg
        # self.triples = triples_pos + triples_neg
        self.triples = triples_pos #存储了所有样本（包括正例和负例）的元组，而 self.questions 存储了所有样本转换为 QA 格式的结果。
        self.len = len(self.triples) #记录了数据集的长度。
        #将正例的数据存储在 self.triples_pos 中，并将总体的长度存储在 self.len 中。

        # convert triple to QA
        questions_pos = [] #存储将正例数据转换为 QA 格式的结果，其中每个问题是一个字典，包含图像路径、操作列表和答案。
        for triple in triples_pos:
            # question = {
            #     "image_id": triple[0],
            #     "op_list": [
            #         {"op": "objects", "param": ""},  # 获取所有物体
            #         {"op": "filter_index", "param": triple[1]},  # 通过filter找到 triple中object的物体
            #         {"op": "relate", "param": id2rel[triple[3]]},  # 以object物体为起点，通过relate操作查询于其具有triple中rel关系的物体
            #         {"op": "filter_index", "param": triple[2]}],  # 通过filter判定该物体是否是triple中subject物体
            #     "answer": triple[2],  # 答案就是triple中的subject
            #     "type": "index"
            # }
            attribute_list=torch.zeros((1,4))
            attribute_list[0][triple[1]]=1#[[1,0,0,0]],意思就是是哪个类别，就在这个列表的哪个位置置为1，标签为0代表第一个位置为1
            attribute_list=torch.reshape(attribute_list, (1, -1))
            # print(" attribute_list:", attribute_list)
            question =[] 
            for i in range(2): #构造测试集的操作列表，由于该属性有两个概念，每个概念要一个操作列表，由于神经网络输出两个概念的值，故每个操作列表的概念矩阵索引分别对应了各自concept的位置
                answer=torch.zeros((1))
                answer[0]=attribute_list[0][i]#[0]or[1]，
                q={
                    "image_path": triple[0],
                    "op_list": [
                        {"op": "objects", "param": ""},  # 获取所有物体
                        {"op": "filter_nearest_obj", "param": ""},  # 找到最近的物体
                        {"op": "obj_attibute", "param":[0,i]}],  # 通过filter判定该物体是否是triple中subject物体
                    "answer": answer,  # 答案就是triple中的subject
                }
                question.append(q)
            questions_pos.append(question)

        #questions_neg = []
        # for triple in triples_neg:
        #     question = {
        #         "image_id": triple[0],
        #         "op_list": [
        #             {"op": "objects", "param": ""},
        #             {"op": "filter_index", "param": triple[1]},
        #             {"op": "relate", "param": id2rel[triple[3]]},
        #             {"op": "filter_index", "param": triple[2]}],
        #         "answer": triple[2],
        #         "type": "index_neg"
        #     }
        #     questions_neg.append(question)

        # self.questions = questions_pos + questions_neg
        self.questions = questions_pos
        self.len = len(self.questions)

    def __getitem__(self, index):
        #根据索引提取数据集中的一个样本，包括图像、两个操作列表和两个答案。返回的元组可以用于模型的训练和评估。
        img_path = self.questions[index][0]['image_path']#取出第一个op_list的图像作为图像

        img_file_path = DATA_INPUT + img_path #构建完整的图像文件路径，将数据文件夹路径 DATA_INPUT 和图像路径拼接起来。
        # ann_file = DATA_INPUT + 'Annotation/shut-' + str(img_id).zfill(3) + '.xml'
        # print(img_file)
        img = imgtool.load_img(img_file_path) #使用 ImageTool 类加载图像，得到图像的张量表示。
        op_lists=[] #初始化一个空列表，用于存储两个操作列表。
        for i in range(2): #循环遍历两个操作列表，并将它们添加到 op_lists 中。
            op_list = self.questions[index][i]['op_list']
            op_lists.append(op_list)
        answers=[] #初始化一个空列表，用于存储两个答案。
        for i in range(2): #循环遍历两个答案，并将它们添加到 answers 中。
            answer =self.questions[index][i]['answer']
            answers.append(answer)

        return (img, op_lists, answers,img_file_path) #返回一个包含图像、两个操作列表、两个答案以及图像文件路径的元组。


    # def __getitem__(self, index):
    #     img_id = self.questions[index]['image_id']
    #     img_id_1 = self.questions[index+1]['image_id']
    #
    #     img_file = DATA_INPUT + 'images/shut-' + str(img_id).zfill(3) + '.jpg'
    #     ann_file = DATA_INPUT + 'Annotation/shut-' + str(img_id).zfill(3) + '.xml'
    #
    #     img_file_1 = DATA_INPUT + 'images/shut-' + str(img_id_1).zfill(3) + '.jpg'
    #     ann_file_1 = DATA_INPUT + 'Annotation/shut-' + str(img_id_1).zfill(3) + '.xml'
    #
    #     # print(img_file)
    #     img, ann = imgtool.load_img(img_file, ann_file)
    #     op_list = self.questions[index]['op_list']
    #     type = self.questions[index]['type']
    #     name_t = 'shut-' + str(img_id).zfill(3)
    #
    #     img_1, ann_1 = imgtool.load_img(img_file_1, ann_file_1)
    #     op_list_1 = self.questions[index+1]['op_list']
    #     type_1 = self.questions[index+1]['type']
    #     name_t_1 = 'shut-' + str(img_id_1).zfill(3)
    #
    #     if type == "index_neg":
    #         answer = torch.zeros(60)  # 构建一个所有物体都不被选中为0的向量
    #     else:
    #         answer = self.eye_matrix[self.questions[index]['answer']]  # 构建一个仅有答案obj为1其他均为0的向量
    #
    #     pic = '/yq/ddd/intel_amm_2/data-end2end-triple/images/shut-' + str(img_id).zfill(3) + '.jpg'
    #     xml_path = '/yq/ddd/intel_amm_2/data-end2end-triple/Annotation/shut-' + str(img_id).zfill(3) + '.xml'
    #     crop_and_filter_objects(pic, xml_path)
    #
    #     return (img, ann, op_list, answer, type, name_t)



    def __len__(self):
        return self.len


if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args() #用于解析已知参数，且出现为识别的参数报错
    # Model Load
    # model_black = Reasoning(args) #初始模型
    # for name, param in model_out.named_parameters(): #named_parameters()函数用于获取模块参数的标准方式，他使得可以方便地访问和操作神经网络模型中的参数。
    #     print(name, param.size(), type(param))
    # model_black = torch.load('./checkpoint/net_in_0.9862.pkl',map_location=torch.device(device)) #导入之前训练好的模型

    # train(model_black) # 训练流程
    
    iii()  # 测试流程