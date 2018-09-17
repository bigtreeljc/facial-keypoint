from term import *
import unittest
import dataset as ds
import model as t
import train
import random
import logging
import tqdm
import time
import torch
import sys
# logging.getLogger().setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

class test(unittest.TestCase):
    def test1(self):
        data_dir = "data"
        loader = ds.kaggle_face_dataset(data_dir, 32)
        training_samples, labels, training_ori, labels_ori = loader.load()
        logging.debug("t_pics {} t_labels {}".format(
            type(training_samples), type(labels)))
        logging.debug("n_pics {} n_labels {}".format(
            len(training_samples), len(labels)))

        n_samples = len(labels)
        sampled_ind = random.randint(0, n_samples)
        logging.debug("showing picture {}".format(sampled_ind))

        # ds.plot_sample(training_ori[sampled_ind], labels_ori[sampled_ind])
        ds.plot_sample_denormed(training_samples[sampled_ind], labels[sampled_ind])

    def test1_1(self):
        data_dir = "data"
        loader = ds.kaggle_face_dataset(data_dir, 32, test=True)
        testing_samples, labels, testing_ori, labels_ori = \
                loader.X, loader.y, loader.X_ori, loader.y_ori
        logging.debug("t_pics {} t_labels {}".format(
            type(testing_samples), type(labels)))
        logging.debug("n_pics {} n_labels {}".format(
            len(testing_samples), 0))

        n_samples = len(testing_samples)
        sampled_ind = random.randint(0, n_samples)
        logging.debug("showing picture {}".format(sampled_ind))

        # ds.plot_sample(training_ori[sampled_ind], labels_ori[sampled_ind])
        ds.plot_testing_sample(testing_samples[sampled_ind])

    def test2(self):
        data_dir = "data"
        loader = ds.kaggle_face_dataset(data_dir, batch_size=32)
        n_iters = 0
        # dataset_iter = iter(loader)
        for batch in iter(loader):
            if batch == StopIteration:
                break
            batch_X, batch_y = batch
            logging.debug("len batch {}".format(len(batch_y)))
            logging.debug("batch shape {}".format(batch_X.shape))
            n_iters += 1
        logging.debug("n_iters actual {} | {}".format(n_iters, len(loader)))

    def test3(self):
        conf_file = "conf.yml"
        conf_o = t.conf(conf_file)
        logging.debug("conf \n{}".format(conf_o))
        logging.debug("\nconf model \n{}\npersist\n{}".format(
            conf_o.model, conf_o.persist))
        logging.debug("learining rate {}".format(
            conf_o.model.learning_rate))

    def test4(self):
        logging.debug("testing network 1 for ""regressive prediction")
        data_dir = "data"
        loader = ds.kaggle_face_dataset(data_dir, batch_size=32)
        first_batch = next(iter(loader))
        batch_X, batch_y = first_batch
        first_batch = torch.from_numpy(batch_X)
        logging.debug("first_batch {}".format(
            first_batch.size()))
        use_cuda = torch.cuda.is_available()
        model1 = t.keypoint_regression_model()
        if use_cuda:
            first_batch = first_batch.cuda()
            model1 = model1.cuda()

        y = model1(first_batch)
        logging.debug("y shape {}".format(y.size()))
        logging.debug("target y shape {}".format(batch_y.shape))

    def test5(self):
        data_dir = "data"
        loader = ds.kaggle_face_dataset(data_dir, batch_size=32)
        i: int = 0
        for i in tqdm.tqdm(range(10000000), 
                desc="{}train_loop iter {}{}".format(
                    Fore.RED, i, Style.RESET_ALL)):
            for batch in tqdm.tqdm(iter(loader), 
                    desc="{}epoch {}{}".format(Fore.BLUE, i, Style.RESET_ALL)):
                if batch == StopIteration:
                    # logging.info("{}done with epoch {}{}".format(
                    #     Back.RED, i, Style.RESET_ALL))
                    logging.info(green("dont with epoch {}".format(cyan(i))))
                    break
                time.sleep(0.02)

    def test6(self):
        print(red_bg(blue("hahaha")))
        print(light_green_bg(light_red("I am god")))
        print(light_green_bg(red("I am god")))
        print(blue("我是李攀"))
        print(light_blue("我是李攀"))
        print(light_white("hello world"))
        print(white("hello world"))

    def test7(self):
        train.main()

if __name__ == "__main__":
    unittest.main()
