import torch
from lib.model_interface import ModelInterface
from rafs.loss import RAFSLoss
from rafs.nets import RAFSGenerator
from lib import utils, checkpoint
import torch.nn.functional as F


class RAFSModel(ModelInterface):
    def initialize_models(self):
        self.G = RAFSGenerator().cuda(self.gpu).train()

    def set_loss_collector(self):
        self._loss_collector = RAFSLoss(self.args)

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
        
        return [self.dict["I_source"], self.dict["I_target"], self.dict["I_swapped"], self.dict["I_st"], F.interpolate(self.dict["M_s"], (256,256)).mean(dim=1, keepdim=True), F.interpolate(self.dict["M_t"], (256,256)).mean(dim=1, keepdim=True)]
        # return [self.dict["I_source"], self.dict["I_target"], self.dict["I_swapped"], self.dict["I_st"], F.interpolate(self.dict["M_s"], (256,256)).mean(dim=1, keepdim=True), F.interpolate(self.dict["M_t"], (256,256)).mean(dim=1, keepdim=True), self.dict["M"], self.dict["fmap0"].mean(dim=1, keepdim=True), self.dict["fmap1"].mean(dim=1, keepdim=True), self.dict["fmap2"].mean(dim=1, keepdim=True), self.dict["fmap3"].mean(dim=1, keepdim=True)]

    def run_G(self):
        I_swapped, I_st, M_s, M_t = self.G(self.dict["I_source"], self.dict["I_target"])
        token_source = self.G.get_token(self.dict["I_source"]).detach()
        token_swapped = self.G.get_token(I_swapped)
        id_swapped = self.G.get_id(I_swapped)
        id_source = self.G.get_id(self.dict["I_source"])

        self.dict["I_swapped"] = I_swapped
        self.dict["I_st"] = I_st
        self.dict["id_source"] = id_source
        self.dict["id_swapped"] = id_swapped
        self.dict["token_source"] = token_source
        self.dict["token_swapped"] = token_swapped
        self.dict["M_s"] = M_s
        self.dict["M_t"] = M_t
        # self.dict["M"] = M
        # self.dict["fmap0"] = F.interpolate(fmaps[0], (256,256))
        # self.dict["fmap1"] = F.interpolate(fmaps[1], (256,256))
        # self.dict["fmap2"] = F.interpolate(fmaps[2], (256,256))
        # self.dict["fmap3"] = F.interpolate(fmaps[3], (256,256))

    def run_D(self):
        pass

    def validation(self, step):
        with torch.no_grad():
            Y = self.G(self.valid_source, self.valid_target)[0]
        return [self.valid_source, self.valid_target, Y]

    @property
    def loss_collector(self):
        return self._loss_collector
    
    # override
    def set_multi_GPU(self):
        utils.setup_ddp(self.gpu, self.args.gpu_num)

        # Data parallelism is required to use multi-GPU
        self.G = torch.nn.parallel.DistributedDataParallel(self.G, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True).module


    # override
    def save_checkpoint(self, global_step):
        """
        Save model and optimizer parameters.
        """
        checkpoint.save_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=global_step)
        
        if self.args.isMaster:
            print(f"\nPretrained parameters are succesively saved in {self.args.save_root}/{self.args.run_id}/ckpt/\n")
    
    # override
    def load_checkpoint(self):
        """
        Load pretrained parameters from checkpoint to the initialized models.
        """

        self.args.global_step = \
        checkpoint.load_checkpoint(self.args, self.G, self.opt_G, "G")

        if self.args.isMaster:
            print(f"Pretrained parameters are succesively loaded from {self.args.save_root}/{self.args.ckpt_id}/ckpt/")

    # override
    def set_optimizers(self):
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.args.lr_G, betas=(self.args.beta1, self.args.beta2))
