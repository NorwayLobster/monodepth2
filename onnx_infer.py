# -*-coding: utf-8 -*-

import os
import sys
from PIL import Image
import onnxruntime
from torchvision import transforms
import onnx


class ONNXModel:
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name, self.input_shape = self.get_input_meta(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("input_meta:{}".format(self.input_shape))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_meta(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        input_shape = []
        for node in onnx_session.get_inputs():
            input_shape.append(node.shape)
            input_name.append(node.name)
        return input_name, input_shape

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        outputs = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return outputs


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    model = ONNXModel("./monodepth2.onnx")

    # image_path = os.path.join(opt.data_path, folder, "image_02/data", "{:010d}.png".format(frame_id))
    # image_path = os.path.join(opt.data_path, folder, "image_02/data", "{:010d}.png".format(frame_id))
    # w.writelines(image_path + '\n')
    image_path = os.path.join(os.getcwd(), "assets/test_image.jpg")
    # Load image and preprocess
    input_image = Image.open(image_path).convert('RGB')
    original_width, original_height = input_image.size

    feed_width, feed_height = model.input_shape[0][2], model.input_shape[0][3]
    input_image = input_image.resize((feed_height, feed_width), Image.LANCZOS)
    # input_tensor = transforms.ToTensor()(input_image).numpy()
    input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).numpy()
    outputs = model.forward(input_tensor)
    pred_depth = outputs[0]
