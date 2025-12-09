import torch
import os
import time
from tqdm import trange, tqdm
import traceback
import numpy as np

from ..utils.utils import load_json, DictAverageMeter, dump_pkl
from ..models.modules.adan import Adan
from ..models.LMDM import LMDM
from ..datasets.s2_dataset_v2 import Stage2Dataset as Stage2DatasetV2
from ..options.option import TrainOptions


class Trainer:
    def __init__(self, opt: TrainOptions):
        self.opt = opt

        print(time.asctime(), '_init_accelerate')
        self._init_accelerate()

        print(time.asctime(), '_init_LMDM')
        self.LMDM = self._init_LMDM()

        print(time.asctime(), '_init_dataset')
        self.data_loader = self._init_dataset()

        print(time.asctime(), '_init_optim')
        self.optim = self._init_optim()

        print(time.asctime(), '_set_accelerate')
        self._set_accelerate()

        print(time.asctime(), '_init_log')
        self._init_log()

    def _init_accelerate(self):
        opt = self.opt
        if opt.use_accelerate:
            from accelerate import Accelerator
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            self.is_main_process = self.accelerator.is_main_process
            self.process_index = self.accelerator.process_index
        else:
            self.accelerator = None
            self.device = 'cuda'
            self.is_main_process = True
            self.process_index = 0

    def _set_accelerate(self):
        if self.accelerator is None:
            return
        
        self.LMDM.use_accelerator(self.accelerator)
        self.optim = self.accelerator.prepare(self.optim)
        self.data_loader = self.accelerator.prepare(self.data_loader)

        self.accelerator.wait_for_everyone()

    def _init_LMDM(self):
        opt = self.opt

        part_w_dict = None
        if opt.part_w_dict_json:
            part_w_dict = load_json(opt.part_w_dict_json)
        dim_ws = None
        if opt.dim_ws_npy:
            dim_ws = np.load(opt.dim_ws_npy)

        lmdm = LMDM(
            motion_feat_dim=opt.motion_feat_dim,
            audio_feat_dim=opt.audio_feat_dim,
            seq_frames=opt.seq_frames,
            part_w_dict=part_w_dict,   # only for train
            checkpoint=opt.checkpoint,
            device=self.device,
            use_last_frame_loss=opt.use_last_frame_loss,
            use_reg_loss=opt.use_reg_loss,
            dim_ws=dim_ws,
        )

        return lmdm

    def _init_dataset(self):
        opt = self.opt

        if opt.dataset_version in ['v2']:
            Stage2Dataset = Stage2DatasetV2
        else:
            raise NotImplementedError()

        dataset = Stage2Dataset(
            data_list_json=opt.data_list_json, 
            seq_len=opt.seq_frames,
            preload=opt.data_preload, 
            cache=opt.data_cache, 
            preload_pkl=opt.data_preload_pkl, 
            motion_feat_dim=opt.motion_feat_dim, 
            motion_feat_start=opt.motion_feat_start,
            motion_feat_offset_dim_se=opt.motion_feat_offset_dim_se,
            use_eye_open=opt.use_eye_open,
            use_eye_ball=opt.use_eye_ball,
            use_emo=opt.use_emo,
            use_sc=opt.use_sc,
            use_last_frame=opt.use_last_frame,
            use_lmk=opt.use_lmk,
            use_cond_end=opt.use_cond_end,
            mtn_mean_var_npy=opt.mtn_mean_var_npy,
            reprepare_idx_map=opt.reprepare_idx_map,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        return data_loader
    
    def _init_optim(self):
        opt = self.opt
        optim = Adan(self.LMDM.model.parameters(), lr=opt.lr, weight_decay=0.02)
        return optim

    def _init_log(self):
        opt = self.opt
        
        experiment_path = os.path.join(opt.experiment_dir, opt.experiment_name)
        self.error_log_path = os.path.join(experiment_path, 'error')
        
        if not self.is_main_process:
            return

        # ckpt
        self.ckpt_path = os.path.join(experiment_path, 'ckpts')
        os.makedirs(self.ckpt_path, exist_ok=True)

        # save opt
        opt_pkl = os.path.join(experiment_path, 'opt.pkl')
        dump_pkl(vars(opt), opt_pkl)

        # loss log
        loss_log = os.path.join(experiment_path, 'loss.log')
        self.loss_logger = open(loss_log, 'a')

        self.ckpt_file_list_for_clear = []

    def _loss_backward(self, loss):
        self.optim.zero_grad()

        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        self.optim.step()

    def _train_one_step(self, data_dict):
        x = data_dict["kp_seq"]             # (B, L, kp_dim)
        cond_frame = data_dict["kp_cond"]   # (B, kp_dim)
        cond = data_dict["aud_cond"]        # (B, L, aud_dim)

        if not self.opt.use_accelerate:
            x = x.to(self.device)
            cond_frame = cond_frame.to(self.device)
            cond = cond.to(self.device)

        loss, loss_dict = self.LMDM.diffusion(
            x, cond_frame, cond, t_override=None
        )

        return loss, loss_dict

    def _train_one_epoch(self):
        data_loader = self.data_loader

        DAM = DictAverageMeter()

        self.LMDM.train()
        self.local_step = 0
        for data_dict in tqdm(data_loader, disable=not self.is_main_process):
            self.global_step += 1
            self.local_step += 1

            loss, loss_dict = self._train_one_step(data_dict)
            self._loss_backward(loss)

            if self.is_main_process:
                loss_dict['total_loss'] = loss
                loss_dict_val = {k: float(v) for k, v in loss_dict.items()}
                DAM.update(loss_dict_val)

        return DAM

    def _show_and_save(self, DAM: DictAverageMeter):
        if not self.is_main_process:
            return
        
        self.LMDM.eval()

        epoch = self.epoch

        # show all loss
        avg_loss_msg = "|"
        for k, v in DAM.average().items():
            avg_loss_msg += " %s: %.6f |" % (k, v)
        msg = f'Epoch: {epoch}, Global_Steps: {self.global_step}, {avg_loss_msg}'
        print(msg, file=self.loss_logger)
        self.loss_logger.flush()

        # save model
        if self.accelerator is not None:
            state_dict = self.accelerator.unwrap_model(self.LMDM.model).state_dict()
        else:
            state_dict = self.LMDM.model.state_dict()

        ckpt = {
            "model_state_dict": state_dict,
        }
        ckpt_p = os.path.join(self.ckpt_path, f"train_{epoch}.pt")
        torch.save(ckpt, ckpt_p)
        tqdm.write(f"[MODEL SAVED at Epoch {epoch}] ({len(self.ckpt_file_list_for_clear)})")
        
        # clear model
        if epoch % self.opt.save_ckpt_freq != 0:
            self.ckpt_file_list_for_clear.append(ckpt_p)

        if len(self.ckpt_file_list_for_clear) > 5:
            _ckpt = self.ckpt_file_list_for_clear.pop(0)
            try:
                os.remove(_ckpt)
            except:
                traceback.print_exc()
                self.ckpt_file_list_for_clear.insert(0, _ckpt)

    def _train_loop(self):
        print(time.asctime(), 'start ...')

        opt = self.opt

        start_epoch = 1
        self.global_step = 0
        self.local_step = 0
        for epoch in trange(start_epoch, opt.epochs + 1, disable=not self.is_main_process):
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            self.epoch = epoch
            DAM = self._train_one_epoch()

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            if self.is_main_process:
                self.LMDM.eval()
                self._show_and_save(DAM)

        print(time.asctime(), 'done.')

    def train_loop(self):
        try:
            self._train_loop()
        except:
            msg = traceback.format_exc()
            error_msg = f'{time.asctime()} \n {msg} \n'
            print(error_msg)
            t = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            logname = f'{t}_rank{self.process_index}_error.log'
            os.makedirs(self.error_log_path, exist_ok=True)
            errorfile = os.path.join(self.error_log_path, logname)
            with open(errorfile, 'a') as f:
                f.write(error_msg)
            print(f'error msg write into {errorfile}')