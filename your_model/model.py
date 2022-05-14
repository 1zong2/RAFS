import torch
from lib import utils
from lib.model_interface import ModelInterface
from your_model.loss import YourModelLoss
from your_model.nets import YourNet
from submodel.discriminator import StarGANv2Discriminator # with BCE


class YourModel(ModelInterface):
    def initialize_models(self):
        self.G = YourNet(arcface_path=self.args.arcface_path).cuda(self.gpu).train()
        self.D = StarGANv2Discriminator().cuda(self.gpu).train()

    def set_loss_collector(self):
        self._loss_collector = YourModelLoss(self.args)

    def train_step(self, global_step):
        # load batch
        I_source, I_target, same_person = self.load_next_batch()
        same_person = same_person.reshape(-1, 1, 1, 1).repeat(1, 3, 256, 256)

        self.dict = {
            "I_source": I_source,
            "I_target": I_target, 
            "same_person": same_person,
        }

        # run G
        self.run_G()

        # update G
        loss_G = self.loss_collector.get_loss_G(self.dict)
        utils.update_net(self.opt_G, loss_G)

        # run D
        self.run_D()

        # update D
        loss_D = self.loss_collector.get_loss_D(self.dict)
        utils.update_net(self.opt_D, loss_D)

        return [self.dict["I_source"], self.dict["I_target"], self.dict["I_swapped"]]

    def run_G(self):
        I_swapped, id_source, attr_target  = self.G(self.dict["I_source"], self.dict["I_target"])
        attr_swapped = self.G.get_attr(I_swapped)
        id_swapped = self.G.get_id(I_swapped)
        d_adv = self.D(I_swapped)

        self.dict["I_swapped"] = I_swapped
        self.dict["attr_target"] = attr_target
        self.dict["attr_swapped"] = attr_swapped
        self.dict["id_source"] = id_source
        self.dict["id_swapped"] = id_swapped
        self.dict["d_adv"] = d_adv

    def run_D(self):
        self.dict["I_source"].requires_grad_()
        d_real = self.D(self.dict["I_source"])
        d_fake = self.D(self.dict["I_swapped"].detach())

        self.dict["d_real"] = d_real
        self.dict["d_fake"] = d_fake

    def validation(self, step):
        with torch.no_grad():
            Y = self.G(self.valid_source, self.valid_target)[0]
        return [self.valid_source, self.valid_target, Y]

    @property
    def loss_collector(self):
        return self._loss_collector
        