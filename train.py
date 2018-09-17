import os

import torch
import torch.nn as nn
from pydoc import locate
import model as m
import dataset as ds
from term import *
import logging

def get_conf():
    conf_file = "conf.yml"
    conf_o = m.conf(conf_file)
    return conf_o

def mk_cuda(torch_object):
    if torch.cuda.is_available():
        return torch_object.cuda()

def get_model(cf):
    if cf.train.train_from == "scratch":
        logging.debug(purple("train from scratch"))
        return mk_cuda(m.keypoint_regression_model(
                    cf.model.width, cf.model.height,
                    cf.model.num_keypoints, cf.model.n_channel))
    else:
        logging.debug(purple("train from checkpoint {}".format(
            cf.train.train_from)))
        return mk_cuda(torch.load(cf.train.train_from))
        
def train_loop(loader, model, cf):
    n_iter = cf.train.num_epochs 
    criterion = locate(cf.train.loss)()
    optimizer = locate(cf.train.optimizer)(
            model.parameters(), lr=cf.train.learning_rate)
    log_loss_every: int = cf.train.log_loss_every
    log_loss_every_epoch: int = cf.train.log_loss_every_epoch
    save_model_every_epoch: int = cf.train.persist.save_model_every_epoch
    save_model_from_epoch: int = cf.train.persist.save_model_from_epoch
    model_dir: str = cf.train.persist.model_path
    train_history_dir: str = cf.train.persist.train_history_path
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(train_history_dir, exist_ok=True)
    cur_epoch: int = 0
    cur_batch: int = 0
    
    
    with std_out_err_redirect_tqdm() as orig_stdout:
        for i in tqdm(range(n_iter), file=orig_stdout, ncols=50,
                dynamic_ncols=False, desc=yellow("trainloop")):
            # for batch in tqdm.tqdm(iter(loader), 
            #         desc=purple("epoch {}".format(i))):
            for batch in iter(loader):
                if batch == StopIteration:
                    if cur_epoch % log_loss_every_epoch == 0:
                        print("{} done {}".format(
                            blue("epoch %d"%i), yellow("loss %.10f"%loss)))
                    if cur_epoch > save_model_from_epoch and \
                            cur_epoch % save_model_every_epoch == 0:
                        model_path = os.path.join(model_dir, 
                                "fkp_epoch{}.pth".format(cur_epoch))
                        print("saving model at {}".format(
                            green(model_path)))
                        torch.save(model, model_path)
                    break

                batch_X, batch_y = batch
                if len(batch_X) == 0:
                    continue
                batch_X = mk_cuda(torch.from_numpy(batch_X))
                batch_y = mk_cuda(torch.from_numpy(batch_y))

                # logging.debug("input X shape {}".format(batch_X.size()))
                predicted_y = model(batch_X)
                loss = criterion(predicted_y, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if cur_batch % log_loss_every == 0:
                #     logging.info("cur_epoch {} loss {}".format(
                #         cyan(cur_epoch), cyan(loss)))

                cur_batch += 1

            cur_epoch += 1

def main():
    cf = get_conf()
    loader = ds.kaggle_face_dataset(cf.dataset.dir, cf.dataset.batch_size)
    model = get_model(cf)
    train_loop(loader, model, cf)

if __name__ == "__main__":
    main()
