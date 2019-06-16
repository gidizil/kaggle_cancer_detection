import numpy as np
from general_utils import HyperParamsConfig

""" ================================== """
""" Class with multiple helper methods """
""" That different models can inherit  """
""" and to better automate the         """
""" procedure of model creation.       """
""" ================================== """


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
        calculate the output size of feature map after a conv2d operation
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
        calculate the output size of feature map after a pooling operation
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
        self.tmp_img_dims = int(H * W)

    def get_final_feature_map_dims(self):
        """Calculates all relevant dimensions to each layer"""
        for k, v in self.net_dict.items():
            if k.startswith('conv'):
                self._get_conv_op_dims(*v)

            elif k.startswith('pool'):
                self._get_conv_op_dims(*v)

        self._flatten_img_dims()


test = ModelUtils(img_dims_in=48,
                  net_architecture_dict={'conv1': ((5, 5), ),
                                         'pool1': ((2, 2), 2),
                                         'conv2': ((5, 5), ),
                                         'pool2': ((2, 2), 2)})
test.get_final_feature_map_dims()
print(test.tmp_img_dims)