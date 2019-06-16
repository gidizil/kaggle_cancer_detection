import numpy as np
from general_utils import HyperParamsConfig

""" =================================== """
""" Class with multiple helper methods  """
""" That different modeles can inherit  """
""" and to better automate to procedure """
""" of model creation.                  """
""" =================================== """


class ModelUtils:
    def __init__(self, img_dims_in, net_architecture_dict):
        self.img_dims_in = img_dims_in
        self.net_dict = net_architecture_dict
        self.h_params_config = HyperParamsConfig()
        self.params_dict = self.h_params_config.params_dict

        self._set_img_dims()
        self.tmp_img_dims = self.img_dims_in

    def _set_img_dims(self):
        """ handle image dims to tuple format. If None - Extract it"""
        if self.img_dims_in is None:
            self.img_dims_in = (self.params_dict['orig_img_size'],
                                self.params_dict['orig_img_size'])

        elif isinstance(self.img_dims_in, int):
            self.img_dims_in = (self.img_dims_in, self.img_dims_in)

        elif isinstance(self.img_dims_in, (list, tuple)):
            pass
        else:
            print('Input should be a tuple or a list')


    def _get_conv_op_dims(self, conv_filter, stride=1, pad=0):
        """
        calculate the output feature map after a conv2d operation
        :param conv_filter: tuple. (h, w)
        :param stride: int. stride size, defaults to 1
        :param pad: int. padding size from each side. Defaults to 0
        :return: tuple (H_out, W_out). dimensions of feature maps after conv2d
        """

        H, W = self.tmp_img_dims
        h, w = conv_filter

        H_out = 1 + (H + 2 * pad - h) / stride
        W_out = 1 + (W + 2 * pad - w) / stride

        self.tmp_img_dims = (H_out, W_out)

    def _get_pool_op_dims(self, pad, stride=1):
        """
        calculate the output feature map after a pooling operation
        :param pad: tuple. (p_h, p_w)
        :param stride: int. stride size, defaults to 1
        :return: tuple (H_out, W_out). dimensions of feature maps after conv2d
        """
        H, W = self.tmp_img_dims
        p_h, p_w = pad

        H_out = 1 + (H - p_h) / stride
        W_out = 1 + (W - p_w) / stride

        self.tmp_img_dims = (H_out, W_out)

    def _flatten_img_dims(self):
        """ flatten 2D image into 1D array"""
        H, W = self.tmp_img_dims
        flat_img = H * W
        return flat_img

    def get_final_feature_map_dims(self):
        """Calculates all relevant dimensions to each layer"""
        for k, v in self.net_dict.items():
            if k.startswith('conv'):
                self._get_conv_op_dims(*v)

            elif k.startswith('pool'):
                self._get_conv_op_dims(*v)

        return int(self._flatten_img_dims())


""" Example for architecture_dict: """
""" {} """
"""
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv1_bn_2d = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 128, 5)
        self.conv2_bn_2d = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*21*21, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        
        
        # 1. Conv part of network
        x = self.conv1(x)
        x = self.conv1_bn_2d(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.conv2_bn_2d(x)
        x = F.relu(x)
        x = self.pool(x)

        # 2. Feed Forward part of network
        x = x.view(-1, 128 * 21 * 21)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""

test = ModelUtils(img_dims_in=48,
                  net_architecture_dict={'conv1': ((5, 5), ),
                                         'pool1': ((2, 2), 2),
                                         'conv2': ((5, 5), ),
                                         'pool2': ((2, 2), 2)})
print(np.sqrt(test.get_final_feature_map_dims()))