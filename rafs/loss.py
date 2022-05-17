from lib.loss_interface import Loss, LossInterface


class RAFSLoss(LossInterface):
    def get_loss_G(self, dict):
        L_G = 0.0
        
        # Id loss
        if self.args.W_id:
            L_id = Loss.get_id_loss(dict["id_source"], dict["id_swapped"])
            L_G += self.args.W_id * L_id
            self.loss_dict["L_id"] = round(L_id.item(), 4)

        # token loss
        if self.args.W_token:
            L_token = Loss.get_L1_loss(dict["token_source"], dict["token_swapped"])
            L_G += self.args.W_token * L_token
            self.loss_dict["L_token"] = round(L_token.item(), 4)

        # Reconstruction loss
        if self.args.W_recon:
            L_recon = Loss.get_L2_loss(dict["I_target"]*dict["same_person"], dict["I_swapped"]*dict["same_person"])
            L_G += self.args.W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)

        # LPIPS loss
        if self.args.W_lpips:
            L_lpips = Loss.get_lpips_loss(dict["I_swapped"], dict["I_target"])
            L_G += self.args.W_lpips * L_lpips
            self.loss_dict["L_lpips"] = round(L_lpips.item(), 4)
                
        self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G

    def get_loss_D(self, dict):
        pass