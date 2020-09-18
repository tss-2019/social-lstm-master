import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
import math
from torch.autograd import Variable
from helper import *


class DataLoader():

    def __init__(self, f_prefix, batch_size=5, seq_length=20, num_of_validation=0, forcePreProcess=False, infer=False,
                 generate=False):
        # DataLoader类的初始化函数
        # batch_size：批处理的大小
        # seq_length：要考虑的序列长度
        # num_of_validation：将使用验证数据集的数量
        # infer：test模式的标志
        # generate：数据生成模式的标志
        # forcePreProcess：标记以强制再次从csv文件中预处理数据

        # base test files
        base_test_dataset = ['/data/test/biwi/biwi_eth.txt',
                             # '/data/test/crowds/crowds_zara01.txt',
                             # '/data/test/crowds/uni_examples.txt',
                             # '/data/test/stanford/coupa_0.txt',
                             # '/data/test/stanford/coupa_1.txt', '/data/test/stanford/gates_2.txt',
                             # '/data/test/stanford/hyang_0.txt', '/data/test/stanford/hyang_1.txt',
                             # '/data/test/stanford/hyang_3.txt', '/data/test/stanford/hyang_8.txt',
                             # '/data/test/stanford/little_0.txt', '/data/test/stanford/little_1.txt',
                             # '/data/test/stanford/little_2.txt', '/data/test/stanford/little_3.txt',
                             # '/data/test/stanford/nexus_5.txt', '/data/test/stanford/nexus_6.txt',
                             # '/data/test/stanford/quad_0.txt', '/data/test/stanford/quad_1.txt',
                             # '/data/test/stanford/quad_2.txt', '/data/test/stanford/quad_3.txt'
                             ]
        # base train files
        base_train_dataset = [# '/data/train/didida/didida2.txt',
                              #  '/data/train/bingchang/111.txt'
                              # '/data/train/bingchang/100.txt'
                              '/data/train/biwi/biwi_hotel.txt',
                                # '/data/train/ceshi/100_1.txt',
                              # '/data/train/crowds/arxiepiskopi1.txt', '/data/train/crowds/crowds_zara02.txt',
                              # '/data/train/crowds/crowds_zara03.txt', '/data/train/crowds/students001.txt',
                              # '/data/train/crowds/students003.txt',
                              # '/data/train/mot/PETS09-S2L1.txt',
                              # '/data/train/stanford/bookstore_0.txt', '/data/train/stanford/bookstore_1.txt',
                              # '/data/train/stanford/bookstore_2.txt', '/data/train/stanford/bookstore_3.txt',
                              # '/data/train/stanford/coupa_3.txt', '/data/train/stanford/deathCircle_0.txt',
                              # '/data/train/stanford/deathCircle_1.txt', '/data/train/stanford/deathCircle_2.txt',
                              # '/data/train/stanford/deathCircle_3.txt',
                              # '/data/train/stanford/deathCircle_4.txt', '/data/train/stanford/gates_0.txt',
                              # '/data/train/stanford/gates_1.txt', '/data/train/stanford/gates_3.txt',
                              # '/data/train/stanford/gates_4.txt', '/data/train/stanford/gates_5.txt',
                              # '/data/train/stanford/gates_6.txt', '/data/train/stanford/gates_7.txt',
                              # '/data/train/stanford/gates_8.txt', '/data/train/stanford/hyang_4.txt',
                              # '/data/train/stanford/hyang_5.txt', '/data/train/stanford/hyang_6.txt',
                              # '/data/train/stanford/hyang_9.txt', '/data/train/stanford/nexus_0.txt',
                              # '/data/train/stanford/nexus_1.txt', '/data/train/stanford/nexus_2.txt',
                              # '/data/train/stanford/nexus_3.txt', '/data/train/stanford/nexus_4.txt',
                              # '/data/train/stanford/nexus_7.txt', '/data/train/stanford/nexus_8.txt',
                              # '/data/train/stanford/nexus_9.txt'
                              ]
        # dimensions of each file set
        # self.dataset_dimensions = {'biwi': [720, 576], 'crowds': [720, 576], 'stanford': [595, 326], 'mot': [768, 576]}
        self.dataset_dimensions = {'train': [595, 326], 'validation': [595, 326]}
        # self.dataset_dimensions = {'test': [720, 576]}

        # 原始数据所在的数据目录列表
        self.base_train_path = 'data/train/'
        self.base_test_path = 'data/test/'
        self.base_validation_path = 'data/validation/'

        # 当infer为True的时候选择test目录为基本目录
        if infer is False:
            self.base_data_dirs = base_train_dataset
        else:
            self.base_data_dirs = base_test_dataset

        # 使用python os和基本目录获取所有文件
        self.train_dataset = self.get_dataset_path(self.base_train_path, f_prefix)
        self.test_dataset = self.get_dataset_path(self.base_test_path, f_prefix)
        self.validation_dataset = self.get_dataset_path(self.base_validation_path, f_prefix)

        # if generate mode, use directly train base files
        if generate:
            self.train_dataset = [os.path.join(f_prefix, dataset[1:]) for dataset in base_train_dataset]

        # request of use of validation dataset
        if num_of_validation > 0:
            self.additional_validation = True
        else:
            self.additional_validation = False

        # 检查验证数据集的可用性，如果大于可用的验证数据集，则裁剪请求的数字
        if self.additional_validation:
            if len(self.validation_dataset) is 0:
                print("没有验证数据集validation.已中止.")
                self.additional_validation = False
            else:
                num_of_validation = np.clip(num_of_validation, 0, len(self.validation_dataset))
                self.validation_dataset = random.sample(self.validation_dataset, num_of_validation)

        # 如果不是infer模式，请使用训练数据集
        if infer is False:
            self.data_dirs = self.train_dataset
        else:
            # 使用validation数据集
            if self.additional_validation:
                self.data_dirs = self.validation_dataset
            # 使用test数据集
            else:
                self.data_dirs = self.test_dataset

        self.infer = infer
        self.generate = generate

        # 数据集数量
        self.numDatasets = len(self.data_dirs)

        # array for keepinng target ped ids for each sequence
        self.target_ids = []

        # Data directory where the pre-processed pickle file resides
        # 预处理数据
        self.train_data_dir = os.path.join(f_prefix, self.base_train_path)
        self.test_data_dir = os.path.join(f_prefix, self.base_test_path)
        self.val_data_dir = os.path.join(f_prefix, self.base_validation_path)

        # 存储参数arguments
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.orig_seq_lenght = seq_length

        # 验证参数arguments
        self.val_fraction = 0

        # 定义存储过程数据的路径
        self.data_file_tr = os.path.join(self.train_data_dir, "trajectories_train.cpkl")
        self.data_file_te = os.path.join(self.base_test_path, "trajectories_test.cpkl")
        self.data_file_vl = os.path.join(self.val_data_dir, "trajectories_val.cpkl")

        #  创建一个字典 key: 文件名, values: 该文件夹中的文件
        self.create_folder_file_dict()

        if self.additional_validation:
            # 如果文件不存在或forcePreProcess为true
            if not (os.path.exists(self.data_file_vl)) or forcePreProcess:
                print("从原始数据创建预处理的验证数据")
                # 预处理数据集中的csv文件中的数据
                # 请注意，此数据以帧为单位进行处理
                self.frame_preprocess(self.validation_dataset, self.data_file_vl, self.additional_validation)

        if self.infer:
            # 如果处于infer模式, and no additional files -> test preprocessing
            if not self.additional_validation:
                if not (os.path.exists(self.data_file_te)) or forcePreProcess:
                    print("从原始数据创建预处理的测试数据")
                    # 预处理数据集中的csv文件中的数据
                    # 请注意，此数据以帧为单位进行处理
                    print("正在处理的文件: ", self.data_file_te)
                    self.frame_preprocess(self.data_dirs, self.data_file_te)
            # 如果处于infer模式, 并且还有其他验证文件 -> validation dataset visualization
            else:
                print("验证可视化文件将被创建")

        # 如果不是infer模式
        else:
            # 如果文件不存在或forcePreProcess为true->训练预处理
            if not (os.path.exists(self.data_file_tr)) or forcePreProcess:
                print("从原始数据创建预处理的训练数据")
                # 预处理数据集中的csv文件中的数据
                # 请注意，此数据以帧为单位进行处理
                self.frame_preprocess(self.data_dirs, self.data_file_tr)

        if self.infer:
            # 从pickle文件中加载处理后的数据
            if not self.additional_validation:  # test mode
                self.load_preprocessed(self.data_file_te)
            else:  # validation mode
                self.load_preprocessed(self.data_file_vl, True)

        else:  # training mode
            self.load_preprocessed(self.data_file_tr)

        # 重置数据加载器对象的所有数据指针
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)

    # 将预处理每个数据集的pixel_pos.csv文件的函数
    # 可以使用占用网格的数据
    def frame_preprocess(self, data_dirs, data_file, validation_set=False):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        validation_set: true when a dataset is in validation set
        '''
        # all_frame_data将是与每个数据集相对应的numpy数组列表的列表
        # 每个numpy数组将对应一个框架，每行的大小为（numPeds，3）
        # 包含pedID，x，y
        all_frame_data = []
        # 验证frame数据
        valid_frame_data = []
        # frameList_data将是与每个数据集相对应的列表的列表
        # 每个列表将包含数据集中所有帧的frameIds
        frameList_data = []
        valid_numPeds_data = []
        # numPeds_data将是与每个数据集相对应的列表的列表
        # 每个列表将包含数据集中每个帧中的行人数量
        numPeds_data = []

        # each list includes ped ids of this frame
        # 每个列表都包含此框架的ped ID
        pedsList_data = []
        valid_pedsList_data = []
        # target ped ids for each sequence
        # 每个序列的目标ped ID
        target_ids = []
        orig_data = []

        # 当前数据集的索引
        dataset_index = 0

        # 对于每个数据集
        for directory in data_dirs:

            # 从txt文件加载数据
            print("Now processing: ", directory)
            column_names = ['frame_num', 'ped_id', 'y', 'x']

            # 如果是训练模式，则将训练文件读取到pandas数据框并进行处理
            if self.infer is False:
                df = pd.read_csv(directory, dtype={'frame_num': 'int', 'ped_id': 'int'}, delimiter=' ', header=None,
                                 names=column_names)
                self.target_ids = np.array(df.drop_duplicates(subset={'ped_id'}, keep='first', inplace=False)['ped_id'])


            else:
                # 如果是验证模式，则将验证文件读取到pandas数据框并进行处理
                if self.additional_validation:
                    df = pd.read_csv(directory, dtype={'frame_num': 'int', 'ped_id': 'int'}, delimiter=' ', header=None,
                                     names=column_names)
                    self.target_ids = np.array(
                        df.drop_duplicates(subset={'ped_id'}, keep='first', inplace=False)['ped_id'])

                # 如果是测试模式，则将测试文件读取到pandas数据框并进行处理
                else:
                    column_names = ['frame_num', 'ped_id', 'y', 'x']
                    df = pd.read_csv(directory, dtype={'frame_num': 'int', 'ped_id': 'int'}, delimiter=' ', header=None,
                                     names=column_names,
                                     converters={c: lambda x: float('nan') if x == '?' else float(x) for c in
                                                 ['y', 'x']})
                    self.target_ids = np.array(
                        df[df['y'].isnull()].drop_duplicates(subset={'ped_id'}, keep='first', inplace=False)['ped_id'])

            # 将pandas数据转化成numpy数组
            data = np.array(df)

            # 保留文件的原始副本
            orig_data.append(data)

            # 交换x和y点（在txt文件中，它就像-> y，x）
            data = np.swapaxes(data, 0, 1)

            # 获取帧号
            frameList = data[0, :].tolist()

            # 帧数
            numFrames = len(frameList)

            # 将frameID列表添加到frameList_data
            frameList_data.append(frameList)
            # 初始化当前数据集的numPeds列表
            numPeds_data.append([])
            valid_numPeds_data.append([])

            # 初始化当前数据集的numpy数组列表
            all_frame_data.append([])
            # 初始化当前数据集的numpy数组列表
            valid_frame_data.append([])

            # peds列表的每一帧
            pedsList_data.append([])
            valid_pedsList_data.append([])

            target_ids.append(self.target_ids)

            for ind, frame in enumerate(frameList):

                # 提取当前帧中的所有行人
                pedsInFrame = data[:, data[0, :] == frame]

                # 提取peds列表
                pedsList = pedsInFrame[1, :].tolist()

                # 将当前帧中的步数添加到存储的数据中

                # 初始化numpy数组的行
                pedsWithPos = []

                # 对于当前帧中的每个ped
                for ped in pedsList:
                    # 提取它们的x和y位置
                    current_x = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[2, pedsInFrame[1, :] == ped][0]

                    # 将其pedID，x，y添加到numpy数组的行中
                    pedsWithPos.append([ped, current_x, current_y])

                # 在推断时，数据生成，如果数据集是验证数据集，则没有验证数据
                if (ind >= numFrames * self.val_fraction) or (self.infer) or (self.generate) or (validation_set):
                    # Add the details of all the peds in the current frame to all_frame_data
                    all_frame_data[dataset_index].append(np.array(pedsWithPos))
                    pedsList_data[dataset_index].append(pedsList)
                    numPeds_data[dataset_index].append(len(pedsList))


                else:
                    valid_frame_data[dataset_index].append(np.array(pedsWithPos))
                    valid_pedsList_data[dataset_index].append(pedsList)
                    valid_numPeds_data[dataset_index].append(len(pedsList))

            dataset_index += 1
        # 将数组保存在pickle文件中
        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numPeds_data, valid_numPeds_data, valid_frame_data, pedsList_data,
                     valid_pedsList_data, target_ids, orig_data), f, protocol=2)
        f.close()

    # 将预处理的数据加载到DataLoader对象中的函数
    # data_file：pickle：数据文件的路径
    # validation_set：验证数据集的标志
    def load_preprocessed(self, data_file, validation_set=False):
        # 从pickled文件读取数据
        if (validation_set):
            print("加载验证数据集: ", data_file)
        else:
            print("加载训练或测试数据集: ", data_file)

        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # 从pickle文件中获取所有数据
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        self.valid_numPedsList = self.raw_data[3]
        self.valid_data = self.raw_data[4]
        self.pedsList = self.raw_data[5]
        self.valid_pedsList = self.raw_data[6]
        self.target_ids = self.raw_data[7]
        self.orig_data = self.raw_data[8]

        counter = 0
        valid_counter = 0
        print('序列大小(frame) ------>', self.seq_length)
        print('batch的大小(frame)--->-', self.batch_size * self.seq_length)

        # 对于每个数据集
        for dataset in range(len(self.data)):
            # 获取当前数据集的frame数据
            all_frame_data = self.data[dataset]
            valid_frame_data = self.valid_data[dataset]
            dataset_name = self.data_dirs[dataset].split('/')[-1]
            # 计算序列数
            num_seq_in_dataset = int(len(all_frame_data) / (self.seq_length))
            num_valid_seq_in_dataset = int(len(valid_frame_data) / (self.seq_length))
            if not validation_set:
                print('来自训练数据集的训练数据(name, # frame, #sequence)--> ', dataset_name, ':',
                      len(all_frame_data), ':', (num_seq_in_dataset))
                print('来自训练数据集的验证数据(name, # frame, #sequence)--> ', dataset_name, ':',
                      len(valid_frame_data), ':', (num_valid_seq_in_dataset))
            else:
                print('来自验证数据集的验证数据(name, # frame, #sequence)--> ', dataset_name, ':',
                      len(all_frame_data), ':', (num_seq_in_dataset))

            # 使用当前数据集中的序列数递增计数器
            counter += num_seq_in_dataset
            valid_counter += num_valid_seq_in_dataset

        # 计算batch次数
        self.num_batches = int(counter / self.batch_size)
        self.valid_num_batches = int(valid_counter / self.batch_size)

        if not validation_set:
            print('总的training batches次数:', self.num_batches)
            print('总的validation batches次数:', self.valid_num_batches)
        else:
            print('总的validation batches:', self.num_batches)

    # Function to get the next batch of points
    def next_batch(self):
        # 源数据
        x_batch = []
        # 目标数据
        y_batch = []
        # 数据集数据
        d = []

        # 每个序列的pedlist
        numPedsList_batch = []

        # 每个序列的pedlist
        PedsList_batch = []

        # 返回target_id
        target_ids = []

        # 迭代指数
        i = 0
        while i < self.batch_size:
            # 提取当前数据集的frame数据
            frame_data = self.data[self.dataset_pointer]
            numPedsList = self.numPedsList[self.dataset_pointer]
            pedsList = self.pedsList[self.dataset_pointer]
            # 获取当前数据集的frame指针
            idx = self.frame_pointer
            # 虽然当前数据集中还剩下seq_length个frame
            if idx + self.seq_length - 1 < len(frame_data):
                # 此序列中的所有数据
                seq_source_frame_data = frame_data[idx:idx + self.seq_length]
                seq_numPedsList = numPedsList[idx:idx + self.seq_length]
                seq_PedsList = pedsList[idx:idx + self.seq_length]
                seq_target_frame_data = frame_data[idx + 1:idx + self.seq_length + 1]

                # 此frame序列中唯一peds的数量
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)
                numPedsList_batch.append(seq_numPedsList)
                PedsList_batch.append(seq_PedsList)
                # 获取序列的正确目标ped ID
                target_ids.append(
                    self.target_ids[self.dataset_pointer][math.floor((self.frame_pointer) / self.seq_length)])
                # print('获取序列的正确目标ped ID')
                # print(target_ids)
                # print(self.dataset_pointer)
                # print(math.floor((self.frame_pointer) / self.seq_length))

                # 这个是原本代码中存在的，用原始数据的时候不会出错，实际上也越界了，但是不会出bug
                # 但是当使用冰场数据的时候会发生索引越界的情况
                self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1

            else:
                # 剩余帧数不足
                # 递增数据集指针并将frame_pointer设置为零
                self.tick_batch_pointer(valid=False)

        return x_batch, y_batch, d, numPedsList_batch, PedsList_batch, target_ids

    # Function to get the next Validation batch of points
    def next_valid_batch(self):
        # 源数据
        x_batch = []
        # 目标数据
        y_batch = []
        # 数据集数据
        d = []

        # 每个序列的pedlist
        numPedsList_batch = []

        # 每个序列的pedlist
        PedsList_batch = []
        target_ids = []

        # 迭代指数
        i = 0
        while i < self.batch_size:
            # 提取当前数据集的frame数据
            frame_data = self.valid_data[self.valid_dataset_pointer]
            numPedsList = self.valid_numPedsList[self.valid_dataset_pointer]
            pedsList = self.valid_pedsList[self.valid_dataset_pointer]

            # 获取当前数据集的frame指针
            idx = self.valid_frame_pointer
            # 虽然当前数据集中还剩下seq_length个frame
            if idx + self.seq_length < len(frame_data):
                # 此序列中的所有数据
                seq_source_frame_data = frame_data[idx:idx + self.seq_length]
                seq_numPedsList = numPedsList[idx:idx + self.seq_length]
                seq_PedsList = pedsList[idx:idx + self.seq_length]
                seq_target_frame_data = frame_data[idx + 1:idx + self.seq_length + 1]

                # 此帧序列中唯一peds的数量
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)
                numPedsList_batch.append(seq_numPedsList)
                PedsList_batch.append(seq_PedsList)
                # 获取序列的正确目标ped ID
                target_ids.append(
                    self.target_ids[self.dataset_pointer][math.floor((self.valid_frame_pointer) / self.seq_length)])
                self.valid_frame_pointer += self.seq_length

                d.append(self.valid_dataset_pointer)
                i += 1

            else:
                # 剩余帧数不足
                # 递增数据集指针并将frame_pointer设置为零
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d, numPedsList_batch, PedsList_batch, target_ids

    #  Advance the dataset pointer
    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''

        if not valid:
            # 转到下一个数据集
            self.dataset_pointer += 1
            # 将当前数据集的帧指针设置为零
            self.frame_pointer = 0
            # 如果所有数据集均已完成，请再次转到第一个
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
            print("*******************")
            print("现在处理: %s" % self.get_file_name())
        else:
            # 转到下一个数据集
            self.valid_dataset_pointer += 1
            # 将当前数据集的帧指针设置为零
            self.valid_frame_pointer = 0
            # 如果所有数据集均已完成，请再次转到第一个
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0
            print("*******************")
            print("现在处理: %s" % self.get_file_name(pointer_type='valid'))

    # 重置所有指针
    def reset_batch_pointer(self, valid=False):
        if not valid:
            # 转到第一个数据集的第一帧
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0

    # 在训练期间用于在训练和验证数据集之间切换的功能
    def switch_to_dataset_type(self, train=False, load_data=True):
        print('--------------------------------------------------------------------------')
        if not train:  # 如果是train模式，请切换至validation模式
            if self.additional_validation:
                print("数据集类型切换: training ----> validation")
                self.orig_seq_lenght, self.seq_length = self.seq_length, self.orig_seq_lenght
                self.data_dirs = self.validation_dataset
                self.numDatasets = len(self.data_dirs)
                if load_data:
                    self.load_preprocessed(self.data_file_vl, True)
                    self.reset_batch_pointer(valid=False)
            else:
                print("没有验证数据集。已中止。")
                return
        else:  # 如果是validation模式，请切换至train模式
            print("数据集类型切换: validation -----> training")
            self.orig_seq_lenght, self.seq_length = self.seq_length, self.orig_seq_lenght
            self.data_dirs = self.train_dataset
            self.numDatasets = len(self.data_dirs)
            if load_data:
                self.load_preprocessed(self.data_file_tr)
                self.reset_batch_pointer(valid=False)
                self.reset_batch_pointer(valid=True)

    # 转换器功能转换为适当的格式。 我们不是直接使用ped ID，而是将ped ID映射到
    # 使用每个表的查找表的数组索引->speed
    # 输出：seq_lenght（长度为实数的序列+1）* max_ped_id + 1（序列中的最大ID号）* 2（x，y）
    def convert_proper_array(self, x_seq, num_pedlist, pedlist):
        # 从序列中获取唯一的ID
        unique_ids = pd.unique(np.concatenate(pedlist).ravel().tolist()).astype(int)
        # 创建一个查找表以映射ped id-> array indices(数组索引)
        lookup_table = dict(zip(unique_ids, range(0, len(unique_ids))))

        seq_data = np.zeros(shape=(self.seq_length, len(lookup_table), 2))

        # 创建数组的新结构
        for ind, frame in enumerate(x_seq):
            corr_index = [lookup_table[x] for x in frame[:, 0]]
            seq_data[ind, corr_index, :] = frame[:, 1:3]

        return_arr = Variable(torch.from_numpy(np.array(seq_data)).float())

        return return_arr, lookup_table

    def add_element_to_dict(self, dict, key, value):
        # helper function to add a element to dictionary
        dict.setdefault(key, [])
        dict[key].append(value)

    def get_dataset_path(self, base_path, f_prefix):
        # get all datasets from given set of directories
        dataset = []
        dir_names = unique_list(self.get_all_directory_namelist())
        for dir_ in dir_names:
            dir_path = os.path.join(f_prefix, base_path, dir_)
            file_names = get_all_file_names(dir_path)
            [dataset.append(os.path.join(dir_path, file_name)) for file_name in file_names]
        return dataset

    def get_file_name(self, offset=0, pointer_type='train'):
        # return file name of processing or pointing by dataset pointer
        if pointer_type is 'train':
            return self.data_dirs[self.dataset_pointer + offset].split('/')[-1]

        elif pointer_type is 'valid':
            return self.data_dirs[self.valid_dataset_pointer + offset].split('/')[-1]

    def create_folder_file_dict(self):
        # create a helper dictionary folder name:file name
        self.folder_file_dict = {}
        for dir_ in self.base_data_dirs:
            folder_name = dir_.split('/')[-2]
            file_name = dir_.split('/')[-1]
            self.add_element_to_dict(self.folder_file_dict, folder_name, file_name)

    def get_directory_name(self, offset=0):
        # return folder name of file of processing or pointing by dataset pointer
        folder_name = self.data_dirs[self.dataset_pointer + offset].split('/')[-2]
        return folder_name

    def get_directory_name_with_pointer(self, pointer_index):
        # get directory name using pointer index
        folder_name = self.data_dirs[pointer_index].split('/')[-2]
        return folder_name

    def get_all_directory_namelist(self):
        # return all directory names in this collection of dataset
        folder_list = [data_dir.split('/')[-2] for data_dir in (self.base_data_dirs)]
        return folder_list

    def get_file_path(self, base, prefix, model_name='', offset=0):
        # return file path of file of processing or pointing by dataset pointer
        folder_name = self.data_dirs[self.dataset_pointer + offset].split('/')[-2]
        base_folder_name = os.path.join(prefix, base, model_name, folder_name)
        return base_folder_name

    def get_base_file_name(self, key):
        # return file name using folder- file dictionary
        return self.folder_file_dict[key]

    def get_len_of_dataset(self):
        # return the number of dataset in the mode
        return len(self.data)

    def clean_test_data(self, x_seq, target_id, obs_lenght, predicted_lenght):
        # remove (pedid, x , y) array if x or y is nan for each frame in observed part (for test mode)
        for frame_num in range(obs_lenght):
            nan_elements_index = np.where(np.isnan(x_seq[frame_num][:, 2]))

            try:
                x_seq[frame_num] = np.delete(x_seq[frame_num], nan_elements_index[0], axis=0)
            except ValueError:
                print("an error has been occured")
                pass

        for frame_num in range(obs_lenght, obs_lenght + predicted_lenght):
            nan_elements_index = x_seq[frame_num][:, 0] != target_id

            try:
                x_seq[frame_num] = x_seq[frame_num][~nan_elements_index]

            except ValueError:
                pass

    def clean_ped_list(self, x_seq, pedlist_seq, target_id, obs_lenght, predicted_lenght):
        # remove peds from pedlist after test cleaning
        target_id_arr = [target_id]
        for frame_num in range(obs_lenght + predicted_lenght):
            pedlist_seq[frame_num] = x_seq[frame_num][:, 0]

    def write_to_file(self, data, base, f_prefix, model_name):
        # write all files as txt format
        self.reset_batch_pointer()
        for file in range(self.numDatasets):
            path = self.get_file_path(f_prefix, base, model_name, file)
            file_name = self.get_file_name(file)
            self.write_dataset(data[file], file_name, path)

    def write_dataset(self, dataset_seq, file_name, path):
        # write a file in txt format
        print("Writing to file  path: %s, file_name: %s" % (path, file_name))
        out = np.concatenate(dataset_seq, axis=0)
        np.savetxt(os.path.join(path, file_name), out, fmt="%1d %1.1f %.3f %.3f", newline='\n')

    # 编写绘图文件以pkl格式进一步可视化
    def write_to_plot_file(self, data, path):
        self.reset_batch_pointer()
        for file in range(self.numDatasets):
            file_name = self.get_file_name(file)
            file_name = file_name.split('.')[0] + '.pkl'
            print("Writing to plot file  path: %s, file_name: %s" % (path, file_name))
            with open(os.path.join(path, file_name), 'wb') as f:
                pickle.dump(data[file], f)

    # 此序列中预测帧号的开始和结束。
    def get_frame_sequence(self, frame_lenght):
        begin_fr = (self.frame_pointer - frame_lenght)
        end_fr = (self.frame_pointer)
        frame_number = self.orig_data[self.dataset_pointer][begin_fr:end_fr, 0].transpose()
        return frame_number

    def get_id_sequence(self, frame_lenght):
        # begin and end of predicted fram numbers in this seq.
        begin_fr = (self.frame_pointer - frame_lenght)
        end_fr = (self.frame_pointer)
        id_number = self.orig_data[self.dataset_pointer][begin_fr:end_fr, 1].transpose()
        id_number = [int(i) for i in id_number]
        return id_number

    def get_dataset_dimension(self, file_name):
        # return dataset dimension using dataset file name
        return self.dataset_dimensions[file_name]
