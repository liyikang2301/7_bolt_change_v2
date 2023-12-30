from email.mime import image
import torch
import numpy as np
import os, json, cv2, random
import xml.etree.ElementTree as ET
import torch.nn as nn
from PIL.ImageDraw import ImageDraw
from PIL.ImageFont import ImageFont
# from timm.data.mixup import Mixup

import time
from crop_pic_sin import crop_and_filter_objects
from models.vit_attribute_model import OnClassify_v3
from models.vit_attribute_model_2 import OnClassify_v3_2
from models.vit_attribute_model_3 import OnClassify_v3_3

from models.models_vit import VisionTransformer

from models.rel_models import OnClassify_v1
from models.engine_finetune import train_one_epoch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
from torchviz import make_dot
from  models import models_vit

max_attribute_num = 3

max_obj_num = 1
max_concept_num = 2
max_rel_num = 10

concept_matrix = None
relate_matrix = None

attribute2id = {'black': 0, 'transp': 1, 'shape':2}
# concept2id_name = {'电流表最小刻度(ammeter_min_scale)': 0, '电流表和电压表指针(pointer)_a': 1,  '电流表(ammeter)': 2, '其他(others)': 3}
concept2id_name = {0: {'notrans': 0,'trans': 1},
                   1: {'noblack': 0,'black': 1},
                   2: {'noellipse': 0,'ellipse':1}
                   }
# concept2id_name = {'bolt': 0,'其他(others)': 1}
rel2id = {'zero': 0, 'unzero': 1}
id2rel = {0: 'zero', 1: 'unzero'}

colors = [[48, 48, 255]]  # 255 48 48 RGB


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


class Reasoning(nn.Module):
    '''
    Reasoning负责整体逻辑规则的顺序计算及逻辑判断
    该类继承自nn.Module
    '''

    def __init__(self, args): #args为参数的意思，应该是选择的模型中包含有的
        super(Reasoning, self).__init__()
        #super() 函数调用Reasoning父类
        self.exec = Executor(args) #创建一个名为 exec 的成员变量，并将其初始化为一个 Executor 类的实例
        self.imgtool = ImageTool()

    def concept_matrix2zero(self):
        self.exec.concept_matrix2zero() #将Executor类中的concept_matrix置零

    def forward(self, dcl_op_list, img,img_file_path, mode):
        '''
        说明：对于每个operation的执行结果，统一都是采用一个一维的行向量进行表示，行向量中每个元素表示其所对应的物体在该步骤之后被选中的概率
        （具体见The neuro-symbolic concept learner, Sec 3.1 Quasi-symbolic program execution部分解释）
        '''

        obj_name_list = []  # 列表包含每张图片中所有物体的name,若为开关柄和基座则正常输入name,其他物体则name置为"其他"
        # for label in ann:
        #     obj_name_list.append(label['name'])
        # print(obj_name_list)
        #forward方法，是PyTorch中nn.Module的一个必须实现的方法。该方法接受输入dcl_op_list、img、img_file_path和mode，并返回模型的输出。
        #obj_name_list是一个空列表，用于存储每张图片中所有物体的名称。

        exec = self.exec #Executor负责具体实现每个operator的运算过程
        exec.init(img,img_file_path, mode) #初始化

        buffer = [] #用于存储每个操作的执行结果
        step = 0 #用于记录执行的步骤
        flag_neg = False #是一个标志变量，用于标记是否出现了错误操作
        for tmp in dcl_op_list:  #遍历dcl_op_list中的每个操作，根据操作类型（op），执行相应的操作，并将结果添加到buffer中
            step += 1
            op = tmp['op']
            param = tmp['param']
            # if(mode == 'infer'):
            #     print('[INFO] current op', op, param)
            if op == 'objects':
                buffer.append(torch.ones(max_obj_num, requires_grad=False))
                # img, ann = self.imgtool.load_img('./data-demo/image/img1.jpg', './data-demo/annotation/img1_annotation.xml')
                # img = self.imgtool.addbox(img, buffer[-1], ann)
                # file = './data-demo/image/out0.jpg'.replace('0', str(step))
                # self.imgtool.save_img(file, img)
                # continue
            elif op == 'filter_nearest_obj':
                buffer.append(exec.filter_nearest_obj(buffer[-1])) #buffer[-1]: 这是列表 buffer 中的最后一个元素。
                #过滤保留了最近的物体，而过滤掉其他物体。
            elif op == 'obj_attibute':
                buffer.append(exec.obj_attibute(buffer[-1], param[0],param[1]))
                #buffer[-1]为之前操作执行后的结果，param[0]: 第一个参数，param[1]: 第二个参数
            elif op == 'attibute2sleeve':
                buffer.append(exec.attibute2sleeve(buffer[-1], param))
            elif op == 'filter_name':
                buffer.append(exec.filter_obj_name(buffer[-1], param))
                # img, ann = self.imgtool.load_img('./data-demo/image/img1.jpg', './data-demo/annotation/img1_annotation.xml')
                # img = self.imgtool.addbox(img, buffer[-1], ann)
                # file = './data-demo/image/out0.jpg'.replace('0', str(step))
                # self.imgtool.save_img(file, img)
            elif op == 'filter_index':
                buffer.append(exec.filter_obj_index(buffer[-1], param))
                # img, ann = self.imgtool.load_img('./data-demo/image/img1.jpg', './data-demo/annotation/img1_annotation.xml')
                # img = self.imgtool.addbox(img, buffer[-1], ann)
                # file = './data-demo/image/out0.jpg'.replace('0', str(step))
                # self.imgtool.save_img(file, img)
            elif op == 'relate':
                buffer.append(exec.relate(buffer[-1], param))
                # img, ann = self.imgtool.load_img('./data-demo/image/img1.jpg', './data-demo/annotation/img1_annotation.xml')
                # img = self.imgtool.addbox(img, buffer[-1], ann)
                # file = './data-demo/image/out0.jpg'.replace('0', str(step))
                # self.imgtool.save_img(file, img)
            elif op == 'count':
                buffer.append(exec.count(buffer[-1]))
            elif op == 'intersect':
                buffer.append(exec.intersect(buffer[-1], buffer[param]))
            elif op == 'union':
                buffer.append(exec.union(buffer[-1], buffer[param]))
            elif op == 'and':
                # print(buffer[-1], )
                # print(buffer)
                buffer.append(exec.and_bool(buffer[-1], buffer[param]))
            elif op == 'or':
                buffer.append(exec.or_bool(buffer[-1], buffer[param]))
            elif op == 'exist':
                # img, ann = self.imgtool.load_img('./data-demo/image/img1.jpg', './data-demo/annotation/img1_annotation.xml')
                # img = self.imgtool.addbox(img, buffer[-1], ann)
                # file = './data-demo/image/out0.jpg'.replace('0', str(step))
                # self.imgtool.save_img(file, img)
                buffer.append(exec.exist(buffer[-1]))
            else:
                print('[!!!ERROR!!!] operator not exist!')
            # if (mode == 'infer'):
            #     # print('[INFO]', op, param,  buffer[-1].data)
            #     print('[INFO]', op, param, end=' ')
            #     step_score = exec.exist(buffer[-1])
            #     if step_score <= 0.5 and flag_neg == False:
            #         print('<---- Bad Step!', end=' ')
            #         flag_neg = True
            #         # print('[INFO] 错误操作', op, param)
            #         # return exec.exist(buffer[-1])
            #     print('')
        answer = buffer[-1]
        # print("answer:",answer)
        return answer


class Predicator(nn.Module):
    '''
    Predicator集成一些谓词及相关计算
    '''

    def __init__(self, args):
        super(Predicator, self).__init__()
        self.net_black = OnClassify_v3(args) #创建了一个名为net_in的成员变量，并将其初始化为一个OnClassify_v3类的实例。args是作为参数传递给OnClassify_v3类的构造函数的。
        self.net_transp = OnClassify_v3_2(args)
        self.net_shape = OnClassify_v3_3(args)
    def attributes_classify(self, img,img_file_path):
        #这段代码的主要作用是将输入的图像通过神经网络模型进行内部属性和外部属性的推断，最后将不同属性的预测结果整合在一起返回。这是一个典型的图像分类和属性预测任务的推断过程。
        ###############问？y_pre_in与y_pre_out代表的内部与外部属性分别代表什么######################
        transform = transforms.Compose([  #创建了一个名为 transform 的图像变换（transform），使用 transforms.Compose 将一系列的图像变换操作组合在一起。
            transforms.Resize((224, 224)), #将图像调整大小到 (224, 224)
            transforms.ToTensor(), #转换为 PyTorch 的张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #对图像进行标准化
            #mean：表示每个通道的均值。在标准化过程中，每个通道的像素值都将减去相应通道的均值
            #std：表示每个通道的标准差。标准差用于将每个通道的像素值除以相应通道的标准差。
            #标准化的过程是通过以下公式实现的：
            #input[channel] = (input[channel] - mean[channel]) / std[channel]
            #该操作将像素值缩小到较小范围，有助于训练神经网络更好地处理图像数据，提高模型训练效果，避免输入数据尺度不同导致的优化问题
        ])
        transform_1 = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        pic_ten_1 = transform_1(img)
        # pic_ten_1.show() #该条代码能够让每训练或者推理一幅图片后显示该张图片
        pic_ten = transform(img) #该行代码将img进行了之前定义的一系列变换，将img大小调整为224，224.且将图像转化为pytorch张量，且标准化
       
        pic_ten = pic_ten.unsqueeze(0)
        # 使用 PyTorch 中的 unsqueeze 方法在第一个维度上添加了一个维度，将其转换为一个包含单个样本的批次。
        #这是因为神经网络的输入通常是批次的形式，即使是单个样本也需要在批次的维度上添加一个维度。
        #因为神经网络的权重矩阵和计算都是针对批次进行的，即模型一次性处理多个样本，从而提高训练效率。
        # 即使在推断阶段，通常也会将单个样本包装成一个批次，以便统一处理。
        #这一行代码的结果是得到了一个形状为 (1, C, H, W) 的张量，其中 C 表示通道数，H 和 W 表示图像的高和宽。

        y_pre_black = self.net_black(pic_ten.to(device, torch.float)) #将预处理的图像进行前向传播。对图像进行内部属性的推断
        #self.net_in 是神经网络模型的一个成员变量，用于对图像进行内部属性的推断。这个推断操作将经过预处理的图像 pic_ten 输入到 net_in 模型中。
        #pic_ten.to(device, torch.float) 将图像张量 pic_ten 移动到指定的设备（通常是 GPU）上，并将数据类型转换为 torch.float。这是为了确保图像张量的数据类型和模型的期望输入一致。
        #self.net_in(...) 实际上是调用神经网络模型 net_in 的前向传播方法。在神经网络中，前向传播方法定义了从输入到输出的计算过程。这个过程包括了图像在网络中的一系列计算，最终产生对内部属性的预测结果。
        #y_pre_in 接收了模型的输出，表示对图像内部属性的预测。这是一个包含概率分布的张量，每个元素表示对应类别的预测概率。
        #总体而言，这行代码实际上是将经过预处理的图像传递给神经网络模型，然后获取模型对图像内部属性的预测结果。这是神经网络在推断阶段的典型操作。
        # max_val,index = torch.max(y_pre_in,dim=1)
        # make_dot(y_pre_in).view()
        y_pre_black = nn.functional.softmax(y_pre_black,dim=1)
        #Softmax函数通常用于将一个实数向量转换成概率分布
        #y_pre_in 是神经网络模型对图像内部属性的原始输出，是一个包含每个类别分数的张量。每个元素表示对应类别的分数，Softmax操作将这些分数转换成概率分布。
        #nn.functional.softmax 是PyTorch中的一个函数，用于进行Softmax操作。接受两个参数，第一个参数为输入张量，第二个是指定要softmax操作在哪个维度进行dim，
        #dim=1表示softmax操作在张量第一个维度进行，通常是对每个类别的分数进行Softmax操作。
        # y_pre_out = self.net_out(pic_ten.to(device, torch.float))
        # y_pre_cor = self.net_cor(pic_ten.to(device, torch.float))
        y_pre_transp = self.net_transp(pic_ten.to(device, torch.float)) #对图像进行外部属性的推断
        #self.net_out 是神经网络模型的一个成员变量，用于对图像进行外部属性的推断。这个推断操作将经过预处理的图像 pic_ten 输入到 net_out 模型中。
        #pic_ten.to(device, torch.float) 将图像张量 pic_ten 移动到指定的设备（通常是 GPU）上，并将数据类型转换为 torch.float。这是为了确保图像张量的数据类型和模型的期望输入一致。
        #self.net_out(...) 实际上是调用神经网络模型 net_out 的前向传播方法。在神经网络中，前向传播方法定义了从输入到输出的计算过程。这个过程包括了图像在网络中的一系列计算，最终产生对外部属性的预测结果。
        #y_pre_out 接收了模型的输出，表示对图像外部属性的预测。这是一个包含概率分布的张量，每个元素表示对应类别的预测概率。
        #总体而言，这行代码实际上是将经过预处理的图像传递给神经网络模型，然后获取模型对图像外部属性的预测结果。这是神经网络在推断阶段的典型操作。
        y_pre_transp = nn.functional.softmax(y_pre_transp,dim=1)
        # y_pre_out_2= torch.zeros((1,2)).to(device, torch.float)
        #创建一个形状为 (1, 2) 的零张量，并将其移动到指定的设备（通常是 GPU），并设置数据类型为 torch.float
        #用于初始化一个变量，以便在后续的计算中进行累积或赋值。

        y_pre_shape = self.net_shape(pic_ten.to(device, torch.float))
        y_pre_shape = nn.functional.softmax(y_pre_shape,dim=1)
        # y_pre_out=torch.concat([y_pre_out_1,y_pre_out_2],axis=0)
        #torch.cat(又称torch.concat)是 PyTorch 中用于拼接张量的函数。它接受一个包含张量的列表（或元组），并指定在哪个维度上进行拼接。
        #axis=0指定在第一个维度拼接，即在行的方向上进行拼接。这意味着两个张量将在行方向上堆叠，形成一个新的张量。
        # y_pre_cor = torch.zeros((1,4)) #创建了一个形状为 (1, 4) 的零张量。这个张量中的所有元素都初始化为零。

        # y_pre[ y_pre>0.5]=1
        # y_pre[ y_pre<0.5]=0

        # loss_1 = y_pre
        # make_dot(loss_1).view()
        # dot = make_dot(loss_1)
        # dot.format = 'png'
        # dot.render(filename='graph_1')
        # max_val = torch.max(y_pre)
        # loss_1 = max_val
        # make_dot(loss_1).view()
        # dot = make_dot(loss_1)
        # dot.format = 'png'
        # dot.render(filename='graph_2')
        # y_pred = torch.tensor([[max_val]])
        # y_pred=torch.cat((y_pre_in, y_pre_out,y_pre_cor),dim=0)
        y_pre_black = torch.reshape(y_pre_black, (1, -1))
        #将 y_pre_in 进行形状重塑，变为一个形状为 (1, -1) 的张量。这里 -1 表示该维度的大小由数据本身和其他维度的大小来自动推断，以确保总的元素数量不变。
        #假设其形状为(M,N),形状重塑的结果是将原始张量展平成一个行向量，即形状为 (1, M*N)。
        y_pre_transp = torch.reshape(y_pre_transp, (1, -1))
        y_pre_shape = torch.reshape(y_pre_shape, (1,-1))
        # y_pre_cor = torch.reshape(y_pre_cor, (1, -1))
        y_pred=[]
        y_pred.append(y_pre_black)
        y_pred.append(y_pre_transp )
        y_pred.append(y_pre_shape)
        # y_pred.append(y_pre_cor)
        # print(y_pred[1])
        ##############y_pre_cor全为0，添加到y_pred中有何作用####################

        # print( y_pred)

        # loss_1 = y_pred
        # make_dot(loss_1).view()
        # dot = make_dot(loss_1)
        # dot.format = 'png'
        # dot.render(filename='graph_3')

        # end_time_classify = time.time()  # 时间计算
        # classify_time = end_time_classify - start_time_classify
        # run_time = str(classify_time)
        # print("classify_time = " + run_time)

        # pred = y_pred.softmax(dim=1)
        return y_pred
        # return y_pred,max_val,index

class ImageTool():
    '''
     Image读取相关方法实现
    '''

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(224),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            transforms.CenterCrop(224),  # 从图片中心切出224*224的图片
            transforms.ToTensor(),  # 将图片(Image)转成Tensor，同时归一化至[0, 1]（直接除以255）
            # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 常用标准化
        ])

    def load_img(self, img_file):
        # img = cv2.imread(img_file)
        img = Image.open(img_file)
        # tmp = PIL.Image.open(img_file)
        # img = self.transform(tmp)

        # image_path, labels = self.parse_xml(anta_file)

        return img

    def parse_xml(self, xml_file):
        """
        解析 xml_文件
        输入：xml文件路径
        返回：图像路径和对应的label信息
        """
        # 使用ElementTree解析xml文件
        tree = ET.parse(xml_file) #使用 ElementTree 模块的 parse 方法解析传入的 XML 文件 (xml_file)。
        root = tree.getroot() #获取了 XML 解析树的根元素。根元素是 XML 文件中所有元素的最顶层元素。
        image_path = '' #存储解析后的图像路径，默认初始化为空字符串。
        labels = [] #存储解析后的标签信息，初始化为空列表。
        DATA_PATH = ''

        for item in root:
            if item.tag == 'filename':
                # 构建图像路径，将文件名拼接到数据路径下的 'VOC2007/JPEGImages' 目录中
                image_path = os.path.join(DATA_PATH, 'VOC2007/JPEGImages', item.text)
            elif item.tag == 'object':
                # 获取对象的名称
                obj_name = item[0].text
                # 将objetc的名称转换为ID
                # obj_num = classes_num[obj_name]
                # 依次得到Bbox的左上和右下点的坐标
                xmin = int(item[4][0].text)
                ymin = int(item[4][1].text)
                xmax = int(item[4][2].text)
                ymax = int(item[4][3].text)

                # if obj_name == '电流表最小刻度(ammeter_min_scale)' or obj_name == '电流表和电压表指针(pointer)_a' or obj_name == '电流表(ammeter)':
                if obj_name == 'bolt':
                    obj_name = obj_name
                else:
                    obj_name = '其他(others)'

                labels.append({'box': [xmin, ymin, xmax, ymax], 'name': obj_name})


        return image_path, labels

    # def img_ann_show(self, img, ann):
    def drawOneBox(img, bbox, color, label):
        '''对于给定的图像与给定的与类别标注信息，在图片上绘制出bbox并且标注上指定的类别
        '''
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(concept2id_name))]
        # 为每个类别创建一个随机颜色
        img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # locate *.ttc
        font = ImageFont.truetype("NotoSansCJK-Bold.ttc", 20, encoding='utf-8')
        #选择了用于标签显示的字体。选择了一个名为 "NotoSansCJK-Bold.ttc" 的字体，大小为 20。

        x1, y1, x2, y2 = bbox
        draw = ImageDraw.Draw(img_PIL) #创建一个可绘制的对象
        position = (x1, y1 - 30) #设置标签的位置
        draw.text(position, label, tuple(color), font=font) #然后通过 draw.text 在指定位置绘制标签。
        img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[0], 2) #cv2.rectangle 用于在图像上绘制矩形，接受参数：图像、矩形的左上角坐标
        # #(x1, y1) 和右下角坐标 (x2, y2)、颜色和线宽度（这里设为 2）。
        return img

    def save_img(self, img_file, img):
        cv2.imwrite(img_file, img)

    def addbox(self, img, buffer, ann):
        #该方法接受图像 (img)、一个缓冲区 (buffer) 以及注释信息 (ann) 作为输入，并在图像上添加边界框。
        #这个方法实现了根据缓冲区值在图像上绘制对象边界框和标签的功能。
        for obj_index in range(min(max_obj_num, len(ann))):
            xmin, ymin, xmax, ymax = ann[obj_index]['box'] #从注释信息中获取当前对象的边界框坐标
            name = ann[obj_index]['name']
            if buffer[obj_index] > 0:
                # cv2.rectangle(img, (x1,y1), (x2, y2), colors[0], 2)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[0], 2)
                #使用 cv2.rectangle 在图像上绘制边界框。(xmin, ymin) 是边界框的左上角坐标，(xmax, ymax) 是右下角坐标，colors[0] 是边界框的颜色，2 是线宽。
                text = str(name) + str(obj_index) + str(buffer[obj_index].data)
                #构建了将要显示在图像上的文本，包括对象的名称、索引和缓冲区的值。
                Font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(img, text, (x,y), Font, Size, (B,G,R), Thickness)
                cv2.putText(img, text, (xmin, ymin - 10), Font, 0.5, colors[0], 1)

        return img


class Executor(nn.Module):
    '''
    Executor负责具体实现每个operator的运算过程
    计算公式具体见concept learner page 18
    '''

    '''
    attribute2id = {'name': 0, 'index': 1, 'balance': 2}
    concept2id_name = {'开关整体(switch)': 0, '开关柄(switch_handle)': 1}
    rel2id = {'up': 0, 'down': 1}
    id2rel = {0: 'up', 1: 'down'}
    '''
    def __init__(self, args):
        super(Executor, self).__init__() #调用了父类 nn.Module 的构造函数，确保正确地初始化了 Executor 类的基类，即 nn.Module。
        self.predicator = Predicator(args) #使得Executor类中可以用self.predicator 来调用 Predicator 类的方法和属性，用于执行预测操作
        self.concept_matrix = None #初值为 None 表示还没有被具体初始化。

        # self._make_concept_matrix(ann)
        # # print(concept_matrix)
        # self._make_relate_matrix(ann)
    

    def init(self,img,img_file_path, mode):
        if  self.concept_matrix == None: #检查 self.concept_matrix 是否为 None，即是否已经被初始化
            self.concept_matrix = self._make_concept_matrix(img,img_file_path, mode)
        # print(concept_matrix)
        # self.relate_matrix = self._make_relate_matrix(ann, name_t, id_a, id_b, mode)
        #实现了对 concept_matrix 属性的初始化。这种设计可以确保在需要使用 concept_matrix 时，它已经被正确初始化。

    def concept_matrix2zero(self):
        self.concept_matrix = None
        #concept_matrix 属性重置为空。这可能在某些情况下用于清空先前的概念矩阵，以便后续重新计算或重新生成新的矩阵。

    def filter_obj_name(self, selected, concept):
        '''
        '''
        concept_idx = concept2id_name[concept] #将 concept 转换为相应的索引 concept_idx。这个字典映射了概念名称到对应的索引。
        mask = self._get_concept_mask(0, concept_idx)  # 0 is attribute name------即为name
        #获取一个表示在给定属性下特定概念的掩码（mask）。
        # mask = torch.min(selected, mask)
        mask = selected * mask
        #这两行将 mask 与外部传入的 selected 张量相乘，实现了对 selected 中的元素进行过滤。最终的 mask 将仅保留那些在指定属性和概念下为真的元素。

        return mask

    def filter_obj_index(self, selected, index):     #这个方法可以不要,index暂时还没实现
        '''
        filter by local index
        '''
        # mask = self._get_concept_mask(1, index)  # 0 is attribute name
        # # mask = torch.min(selected, mask)
        # mask = selected * mask
        mask = torch.zeros(max_obj_num, requires_grad=False)
        mask[index] = 1
        # mask = torch.min(selected, mask)
        mask = selected * mask

        return mask

    def relate(self, selected, rel):
        #根据给定的关系名称，通过关系掩码对 selected 张量进行过滤，得到一个表示满足指定关系的元素的新掩码。
        '''

        '''
        rel_idx = rel2id[rel] #这一行通过从外部引入的 rel2id 字典，将关系名称 rel 转换为相应的索引 rel_idx。这个字典映射了关系名称到对应的索引。
        mask = self._get_relate_mask(rel_idx) #这一行调用了类内部的 _get_relate_mask 方法，传递了 rel_idx 作为参数。这个方法的作用是获取一个表示给定关系的掩码。
        mask = (mask * selected.unsqueeze(-1)).sum(dim=-2)
        #这一行通过将 mask 与 selected 张量进行逐元素相乘，并对结果进行求和，得到了一个关系过滤后的掩码。unsqueeze(-1) 操作在 selected 张量的最后一维添加了一个维度，以便在乘法运算中进行广播。
        # mask = torch.unsqueeze(selected, -1).sum(-2)
        return mask

    def filter_nearest_obj(self, selected):
        #创建 selected 的一个副本而不改变其值。
        '''
        '''
        mask = selected * 1
        return mask
    def obj_attibute(self, selected,attribute_index,concept_index):
        #这个操作可能用于选择特定属性和概念条件下的物体。
        '''
        '''
        # attibute_vec=torch.zeros(4)
        # i=0
        # for attritube_index in range(min(max_attribute_num, len(attribute2id))):
        #     for concept_index in range(min(max_concept_num, len(concept2id_name[attritube_index]))):
        #         attibute_vec[i]= self._get_concept_obj_mask(attritube_index,concept_index,selected)
        #         i+=1
        
        # for con_index in range(4):
        #         attibute_vec[i]= self._get_concept_obj_mask(0,con_index,selected)
        #         i+=1
        # attibute_vec=torch.reshape(attibute_vec, (1, -1))

        mask= self._get_concept_mask(attribute_index,concept_index)
        #接受 attribute_index 和 concept_index 作为参数，返回一个张量 mask。这个 mask 表示在给定属性和概念的条件下，被选中物体的位置。
        mask = selected * mask
        # attibute_vec=torch.zeros(4)
        # attibute_vec[concept_index]=mask[0].data
        # attibute_vec=torch.reshape(attibute_vec, (1, -1))
        return mask
    def exist(self, selected):
        '''
        总体而言，exist 方法的作用是判断输入张量 selected 中是否存在非零元素。如果张量中存在非零元素，selected.max()
        的结果将大于零，因此返回 True；否则，返回 False。这个方法可能用于判断在某些条件下是否存在符合条件的物体。
        '''
        return selected.max()

    def count(self, selected):
        '''
        因此，count 方法的作用是计算输入张量 selected 中非零元素的个数。由于在 PyTorch 中，
        非零元素的值被视为 True，零值被视为 False，因此 selected.sum() 实际上是计算了张量中值为 True
        的元素的个数。这个方法可能用于统计符合某些条件的物体的数量。
        '''
        return selected.sum()

    def and_bool(self, selected1, selected2):
        '''
        对两个输入张量 selected1 和 selected2 进行逻辑 AND 操作，并返回一个新的张量，
        其中每个位置上的元素都是两个输入张量对应位置上元素的最小值。
        '''
        # print(selected1, selected2)
        return torch.min(selected1, selected2)

    def or_bool(self, selected1, selected2):
        '''
        两个输入张量 selected1 和 selected2 进行逻辑 OR 操作，并返回一个新的张量，其中每个位置上的元素都是两个
        输入张量对应位置上元素的最大值。
        如果两个张量中有一个位置上的值为 1（True），那么结果张量的相应位置将为 1；
        只有当两个输入张量在相应位置上的值都为 0（False）时，结果张量的相应位置才为 0。
        '''
        return torch.max(selected1, selected2)

    def intersect(self, selected1, selected2):
        return torch.min(selected1, selected2)
        #只有当两个输入张量在相应位置上的值都为 1（True）时，结果张量的相应位置才为 1；
        # 在其他情况下，结果张量的相应位置为 0（False）
    def union(self, selected1, selected2):
        return torch.max(selected1, selected2)
        #如果 selected1 和 selected2 中的相应元素表示布尔值（0 或 1），那么 torch.max(selected1, selected2) 将执行逻辑 OR 操作。
        # 只要两个输入张量在相应位置上的值之一为 1（True），
        # 结果张量的相应位置就为 1；只有当两个输入张量在相应位置上的值都为 0（False）时，结果张量的相应位置才为 0。

    def _make_concept_matrix(self, img,img_file_path, mode):
        # global concept_matrix
        #创建概念矩阵
        concept_matrix = torch.zeros((max_attribute_num, max_concept_num, max_obj_num), requires_grad=False)   #max_obj_num=60
        #这行代码创建了一个全零的三维张量（tensor）concept_matrix，其形状是 (max_attribute_num, max_concept_num, max_obj_num)。
        #####################问：max_attribute_num=3, max_concept_num=4, max_obj_num=1分别代表什么##################
        # 0 dim is for 'name' concept
        index=0
        res= self.predicator.attributes_classify(img,img_file_path) ## 使用预测器（Predicator）对图像进行属性分类
        # print(res)
        # print(len(attribute2id))
        # print(len(concept2id_name[0]))
        for attritube_index in range(min(max_attribute_num, len(attribute2id))):
            #遍历属性的索引，其中 min(max_attribute_num, len(attribute2id)) 确保不超过最大属性数。
            #min()函数返回两个的最小值
            for concept_index in range(min(max_concept_num, len(concept2id_name[attritube_index]))):
                #遍历概念的索引，其中 min(max_concept_num, len(concept2id_name[attritube_index])) 确保不超过最大概念数。
                for obj_index in range(max_obj_num):
                    # print(ann[obj_index]['name'], concept2id_name[ann[obj_index]['name']])
                    # print(ann[obj_index]['name'])
                    ## 如果处于测试模式
                    if mode == 'test':
                        ## 如果预测值大于等于0.5，将概念矩阵中相应位置设置为1，否则设置为0
                        # res = res.argmax(dim=1)
                        # relate_matrix[0][a_index][b_index] = res[0][0]
                        if (res[attritube_index][0][concept_index].data >= 0.5):
                        # if concept_index==index[0].data and res[attritube_index][concept_index].data >= 0.01:
                            concept_matrix[attritube_index][concept_index][obj_index] = 1 # [a_index][b_index]:a_index索引物体相对于b_index索引物体为up关系
                        else:
                            concept_matrix[attritube_index][concept_index][obj_index] = 0

                    # 如果处于训练或推理模式
                    elif mode == 'train' or mode == 'infer':
                        # 将概念矩阵中相应位置设置为属性分类的结果
                        concept_matrix[attritube_index][concept_index][obj_index] = res[attritube_index][0][concept_index]
                    # if concept2id_name[ann[obj_index]['name']] == concept_index:
                    #     concept_matrix[0][concept_index][obj_index] = 1
        # 1 dim for 'obj local index'
        # concept_matrix[1] = torch.eye(max_obj_num)

        # # 2 dim is for 'balance' concept
        # for obj_index in range(min(max_obj_num, len(ann))):
        #     concept_matrix[2][0][obj_index] = 1
        # print("concept_matrix:",concept_matrix)

        return concept_matrix


    def _get_concept_mask(self, attribute, concept_index):  # (0, concept_index)
        '''

        '''
        return self.concept_matrix[attribute][concept_index]  # 返回的就是这种物体在该图片中出现的情况,如(0, 1, 1)即为图片中共3个物体,第2,3个为此物体
    def _get_concept_obj_mask(self, attribute, concept_index,obj_index):  
        '''

        '''
        return self.concept_matrix[attribute][concept_index][obj_index]

    def _get_relate_mask(self, relate_index):
        return self.relate_matrix[relate_index]
