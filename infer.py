import numpy as np
import random
import dataset as ds
import train as t
import model as m
import torch
import argparse
from term import *

def setup_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-model', type=str, required=True, dest="model_path",
                        help='model path to infer')
    # parser.add_argument('-', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')
    
    return parser.parse_args()

def load_model(model_path):
    # model = m.keypoint_regression_model()
    model = torch.load(model_path)
    return model

def main():
    args = setup_args()
    print(red("args {}".format(args)))

    model = load_model(args.model_path)

    '''
        get a random test picture and imshow the predicted kps
    '''
    data_dir = "data"
    loader = ds.kaggle_face_dataset(data_dir, 32, test=True)
    testing_samples, labels, testing_ori, labels_ori = \
            loader.X, loader.y, loader.X_ori, loader.y_ori
    n_samples = len(testing_samples)
    sampled_ind = random.randint(0, n_samples)
    selected_pic = loader.X[sampled_ind]
    # ds.plot_testing_sample(selected_pic)
    print("selected pic shape {}".format(selected_pic.shape))
    input_tensor = t.mk_cuda(torch.from_numpy(selected_pic.reshape((1, 1, 96, 96))))
    predicted_y = model(input_tensor)
    predicted_y = predicted_y.cpu().detach().numpy().reshape(30)
    print("predicted {}".format(predicted_y))
    ds.plot_sample_denormed(selected_pic, predicted_y)

if __name__ == "__main__":
    main()
