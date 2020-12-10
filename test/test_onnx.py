from torchvision import transforms
import cv2
import numpy as np
import onnxruntime

if __name__ == '__main__':
    # input transform
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    # read input image. As we need aligned face images, all the input image should be aligned with face detection method.
    img = cv2.imread(r'D:\thz\data\RAF-DB\Image\aligned\test_0010_aligned.jpg')
    # transform input image
    img = data_transforms_val(img)
    img = np.expand_dims(img, axis=0)
    # load model
    ort_session = onnxruntime.InferenceSession("emotion_recognition.onnx")
    # inference
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    # get output
    ort_outs = ort_session.run(None, ort_inputs)
    print(np.argmax(ort_outs[1]))