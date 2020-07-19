# Created by Haiping Lu from modifying https://github.com/HaozhiQi/ISONet/blob/master/isonet/trainer.py
# Under the MIT License
import time
import torch
import torch.nn as nn
from kale.utils.print import tprint, pprint_without_newline
from config import C


class Trainer(object):
    def __init__(self, device, train_loader, val_loader, model, optim, logger, output_dir):
        # misc
        self.device = device
        self.output_dir = output_dir
        # data loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        # nn setting
        self.model = model
        self.optim = optim
        # lr setting
        self.criterion = nn.CrossEntropyLoss()
        # training loop settings
        self.epochs = 1
        # loss settings
        self.train_acc, self.val_acc = [], []
        self.best_valid_acc = 0
        self.ce_loss, self.ortho_loss = 0, 0
        # others
        self.ave_time = 0
        self.logger = logger

    def train(self):
        while self.epochs <= C.SOLVER.MAX_EPOCHS:
            self.adjust_learning_rate()
            self.train_epoch()
            self.val()
            self.epochs += 1

    def train_epoch(self):
        self.model.train()
        self.ce_loss = 0
        self.ortho_loss = 0
        self.ave_time = 0
        correct = 0
        total = 0
        epoch_t = time.time()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            iter_t = time.time()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optim.zero_grad()
            batch_size = inputs.shape[0]

            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            loss.backward()
            self.optim.step()

            _, predicted = outputs.max(1)
            total += batch_size

            correct += predicted.eq(targets).sum().item()

            self.ave_time += time.time() - iter_t
            tprint(f'train Epoch: {self.epochs} | {batch_idx + 1} / {len(self.train_loader)} | '
                   f'Acc: {100. * correct / total:.3f} | CE: {self.ce_loss / (batch_idx + 1):.3f} | '
                   f'O: {self.ortho_loss / (batch_idx + 1):.3f} | time: {self.ave_time / (batch_idx + 1):.3f}s')

        info_str = f'train Epoch: {self.epochs} | Acc: {100. * correct / total:.3f} | ' \
                   f'CE: {self.ce_loss / (batch_idx + 1):.3f} | ' \
                   f'time: {time.time() - epoch_t:.2f}s |'
        self.logger.info(info_str)
        pprint_without_newline(info_str)
        self.train_acc.append(100. * correct / total)


# def train(config):
#     ## set pre-process
#     prep_dict = {}
#     prep_config = config["prep"]
#     prep_dict["source"] = prep.image_train(**config["prep"]['params'])
#     prep_dict["target"] = prep.image_train(**config["prep"]['params'])
#     if prep_config["test_10crop"]:
#         prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
#     else:
#         prep_dict["test"] = prep.image_test(**config["prep"]['params'])

#     ## prepare data
#     dsets = {}
#     dset_loaders = {}
#     data_config = config["data"]
#     train_bs = data_config["source"]["batch_size"]
#     test_bs = data_config["test"]["batch_size"]
#     dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
#                                 transform=prep_dict["source"])
#     dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
#             shuffle=True, num_workers=4, drop_last=True)
#     dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
#                                 transform=prep_dict["target"])
#     dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
#             shuffle=True, num_workers=4, drop_last=True)

#     if prep_config["test_10crop"]:
#         for i in range(10):
#             dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
#                                 transform=prep_dict["test"][i]) for i in range(10)]
#             dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
#                                 shuffle=False, num_workers=4) for dset in dsets['test']]
#     else:
#         dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
#                                 transform=prep_dict["test"])
#         dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
#                                 shuffle=False, num_workers=4)

#     class_num = config["network"]["params"]["class_num"]

#     ## set base network
#     net_config = config["network"]
#     base_network = net_config["name"](**net_config["params"])
#     base_network = base_network.cuda()

#     ## add additional network for some methods
#     if config["loss"]["random"]:
#         random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
#         ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
#     else:
#         random_layer = None
#         ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
#     if config["loss"]["random"]:
#         random_layer.cuda()
#     ad_net = ad_net.cuda()
#     parameter_list = base_network.get_parameters() + ad_net.get_parameters()
 
#     ## set optimizer
#     optimizer_config = config["optimizer"]
#     optimizer = optimizer_config["type"](parameter_list, \
#                     **(optimizer_config["optim_params"]))
#     param_lr = []
#     for param_group in optimizer.param_groups:
#         param_lr.append(param_group["lr"])
#     schedule_param = optimizer_config["lr_param"]
#     lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

#     gpus = config['gpu'].split(',')
#     if len(gpus) > 1:
#         ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
#         base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        

#     ## train   
#     len_train_source = len(dset_loaders["source"])
#     len_train_target = len(dset_loaders["target"])
#     transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
#     best_acc = 0.0
#     for i in range(config["num_iterations"]):
#         if i % config["test_interval"] == config["test_interval"] - 1:
#             base_network.train(False)
#             temp_acc = image_classification_test(dset_loaders, \
#                 base_network, test_10crop=prep_config["test_10crop"])
#             temp_model = nn.Sequential(base_network)
#             if temp_acc > best_acc:
#                 best_acc = temp_acc
#                 best_model = temp_model
#             log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
#             config["out_file"].write(log_str+"\n")
#             config["out_file"].flush()
#             print(log_str)
#         if i % config["snapshot_interval"] == 0:
#             torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
#                 "iter_{:05d}_model.pth.tar".format(i)))

#         loss_params = config["loss"]                  
#         ## train one iter
#         base_network.train(True)
#         ad_net.train(True)
#         optimizer = lr_scheduler(optimizer, i, **schedule_param)
#         optimizer.zero_grad()
#         if i % len_train_source == 0:
#             iter_source = iter(dset_loaders["source"])
#         if i % len_train_target == 0:
#             iter_target = iter(dset_loaders["target"])
#         inputs_source, labels_source = iter_source.next()
#         inputs_target, labels_target = iter_target.next()
#         inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
#         features_source, outputs_source = base_network(inputs_source)
#         features_target, outputs_target = base_network(inputs_target)
#         features = torch.cat((features_source, features_target), dim=0)
#         outputs = torch.cat((outputs_source, outputs_target), dim=0)
#         softmax_out = nn.Softmax(dim=1)(outputs)
#         if config['method'] == 'CDAN+E':           
#             entropy = loss.Entropy(softmax_out)
#             transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
#         elif config['method']  == 'CDAN':
#             transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
#         elif config['method']  == 'DANN':
#             transfer_loss = loss.DANN(features, ad_net)
#         else:
#             raise ValueError('Method cannot be recognized.')
#         classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
#         total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
#         total_loss.backward()
#         optimizer.step()
#     torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
#     return best_acc


    def val(self, loader, model, test_10crop=True):
        self.model.eval()
        self.ce_loss = 0
        self.ortho_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        if 100. * correct / total > self.best_valid_acc:
            self.snapshot('best')
        self.snapshot('latest')
        self.best_valid_acc = max(self.best_valid_acc, 100. * correct / total)
        info_str = f'valid | Acc: {100. * correct / total:.3f} | ' \
                   f'CE: {self.ce_loss / len(self.val_loader):.3f} | ' \
                   f'O: {self.ortho_loss / len(self.val_loader):.3f} | ' \
                   f'best: {self.best_valid_acc:.3f} | '
        print(info_str)
        self.logger.info(info_str)
        self.val_acc.append(100. * correct / total)


    # start_test = True
    # with torch.no_grad():
    #     if test_10crop:
    #         iter_test = [iter(loader['test'][i]) for i in range(10)]
    #         for i in range(len(loader['test'][0])):
    #             data = [iter_test[j].next() for j in range(10)]
    #             inputs = [data[j][0] for j in range(10)]
    #             labels = data[0][1]
    #             for j in range(10):
    #                 inputs[j] = inputs[j].cuda()
    #             labels = labels
    #             outputs = []
    #             for j in range(10):
    #                 _, predict_out = model(inputs[j])
    #                 outputs.append(nn.Softmax(dim=1)(predict_out))
    #             outputs = sum(outputs)
    #             if start_test:
    #                 all_output = outputs.float().cpu()
    #                 all_label = labels.float()
    #                 start_test = False
    #             else:
    #                 all_output = torch.cat((all_output, outputs.float().cpu()), 0)
    #                 all_label = torch.cat((all_label, labels.float()), 0)
    #     else:
    #         iter_test = iter(loader["test"])
    #         for i in range(len(loader['test'])):
    #             data = iter_test.next()
    #             inputs = data[0]
    #             labels = data[1]
    #             inputs = inputs.cuda()
    #             labels = labels.cuda()
    #             _, outputs = model(inputs)
    #             if start_test:
    #                 all_output = outputs.float().cpu()
    #                 all_label = labels.float()
    #                 start_test = False
    #             else:
    #                 all_output = torch.cat((all_output, outputs.float().cpu()), 0)
    #                 all_label = torch.cat((all_label, labels.float()), 0)
    # _, predict = torch.max(all_output, 1)
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # return accuracy




    def loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        self.ce_loss += loss.item()

        if C.ISON.ORTHO_COEFF > 0:
            o_loss = self.model.module.ortho(self.device)
            self.ortho_loss += o_loss.item()
            loss += o_loss * C.ISON.ORTHO_COEFF
        return loss

    def adjust_learning_rate(self):
        # if do linear warmup
        if C.SOLVER.WARMUP and self.epochs < C.SOLVER.WARMUP_EPOCH:
            lr = C.SOLVER.BASE_LR * self.epochs / C.SOLVER.WARMUP_EPOCH
        else:
            # normal (step) scheduling
            lr = C.SOLVER.BASE_LR
            for m_epoch in C.SOLVER.LR_MILESTONES:
                if self.epochs > m_epoch:
                    lr *= C.SOLVER.LR_GAMMA

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
            if 'scaling' in param_group:
                param_group['lr'] *= param_group['scaling']

    def snapshot(self, name=None):
        state = {
            'net': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'epoch': self.epochs,
            'train_accuracy': self.train_acc,
            'test_accuracy': self.val_acc
        }
        if name is None:
            torch.save(state, f'{self.output_dir}/{self.epochs}.pt')
        else:
            torch.save(state, f'{self.output_dir}/{name}.pt')
