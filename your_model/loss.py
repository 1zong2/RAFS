from lib.loss_interface import Loss, LossInterface


class YourModelLoss(LossInterface):
    def get_loss_G(self, dict):
        L_G = 0.0
        
        # Adversarial loss
        if self.args.W_adv:
            L_adv = Loss.get_BCE_loss(dict["d_adv"], True)
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_adv"] = round(L_adv.item(), 4)

        # Id loss
        if self.args.W_id:
            L_id = Loss.get_id_loss(dict["id_source"], dict["id_swapped"])
            L_G += self.args.W_id * L_id
            self.loss_dict["L_id"] = round(L_id.item(), 4)

        # Attribute loss
        if self.args.W_attr:
            L_attr = Loss.get_attr_loss(dict["attr_target"], dict["attr_swapped"], self.args.batch_per_gpu)
            L_G += self.args.W_attr * L_attr
            self.loss_dict["L_attr"] = round(L_attr.item(), 4)
            
        # Reconstruction loss
        if self.args.W_recon:
            L_recon = Loss.get_L2_loss(dict["I_target"]*dict["same_person"], dict["I_swapped"]*dict["same_person"])
            L_G += self.args.W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)
        
        # Cycle loss
        if self.args.W_cycle:
            L_cycle = Loss.get_L2_loss(dict["I_target"], dict["I_cycle"])
            L_G += self.args.W_cycle * L_cycle
            self.loss_dict["L_cycle"] = round(L_cycle.item(), 4)


        self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G

    def get_loss_D(self, dict):
        L_real = Loss.get_BCE_loss(dict["d_real"], True)
        L_fake = Loss.get_BCE_loss(dict["d_fake"], False)
        L_reg = Loss.get_r1_reg(dict["d_real"], dict["I_source"])
        L_D = L_real + L_fake + L_reg
        
        self.loss_dict["L_real"] = round(L_real.mean().item(), 4)
        self.loss_dict["L_fake"] = round(L_fake.mean().item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D
        