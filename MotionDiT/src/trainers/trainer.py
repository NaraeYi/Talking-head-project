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
# from stream_pipeline_offline import StreamSDK

import sys
import os

# epoch sample: import StreamSDK lazily so inference/CUDA helpers do not run before DDP setup.
STREAM_SDK_AVAILABLE = None
StreamSDK = None
import librosa
import math

# Try to import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")
    wandb = None  # placeholder

import datetime
from datetime import timezone, timedelta  # 서버시간대X, 한국시간대


def _get_stream_sdk():
    # epoch sample: keep the heavy inference stack out of distributed initialization.
    global STREAM_SDK_AVAILABLE, StreamSDK
    if STREAM_SDK_AVAILABLE is not None:
        return StreamSDK if STREAM_SDK_AVAILABLE else None

    try:
        # trainer.py is at: MotionDiT/src/trainers/trainer.py
        # stream_pipeline_offline.py is at: ditto-talkinghead-train/stream_pipeline_offline.py
        ditto_train_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if ditto_train_dir not in sys.path:
            sys.path.insert(0, ditto_train_dir)
        from stream_pipeline_offline import StreamSDK as _StreamSDK
        StreamSDK = _StreamSDK
        STREAM_SDK_AVAILABLE = True
        print("[INFO] StreamSDK imported successfully for sample generation")
    except ImportError as e:
        STREAM_SDK_AVAILABLE = False
        StreamSDK = None
        print(f"[WARNING] StreamSDK not available - sample video generation disabled: {e}")
    return StreamSDK


class Trainer:
    def __init__(self, opt: TrainOptions):
        self.opt = opt
        self.use_meanflow = getattr(opt, "use_meanflow", False)
        self.meanflow_mode = getattr(opt, "meanflow_mode", "improved")

        print(time.asctime(), '_init_accelerate')
        self._init_accelerate()

        print(time.asctime(), '_init_LMDM / MeanFlow')
        self.LMDM = self._init_LMDM()
        self._configure_pose_branch_training()

        print(time.asctime(), '_init_dataset')
        self.data_loader = self._init_dataset()
        # dyn loss: keep a global total-step count ready for optional loss scheduling.
        self.total_steps = max(1, int(self.opt.epochs) * len(self.data_loader))

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
            precision = getattr(opt, 'precision', 'bf16')
            self.accelerator = Accelerator(split_batches=True, mixed_precision=precision)
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

        # accelerate ddp: train through the diffusion wrapper, so prepare it with optimizer/dataloader together.
        self.LMDM.diffusion, self.optim, self.data_loader = self.accelerator.prepare(
            self.LMDM.diffusion,
            self.optim,
            self.data_loader,
        )
        self.LMDM.model = self.accelerator.unwrap_model(self.LMDM.diffusion).model
        self.LMDM.device = self.device

        self.accelerator.wait_for_everyone()

    def _init_LMDM(self):
        opt = self.opt

        part_w_dict = None
        if opt.part_w_dict_json:
            part_w_dict = load_json(opt.part_w_dict_json)
        dim_ws = None
        if opt.dim_ws_npy:
            dim_ws = np.load(opt.dim_ws_npy)
        # accelerate ddp: build on CPU and let Accelerator place the module on each rank.
        lmdm_device = "cpu" if opt.use_accelerate else self.device

        lmdm = LMDM(
                motion_feat_dim=opt.motion_feat_dim,
                audio_feat_dim=opt.audio_feat_dim,
                seq_frames=opt.seq_frames,
                part_w_dict=part_w_dict,   # only for train
                checkpoint=opt.checkpoint,
                device=lmdm_device,
                use_last_frame_loss=opt.use_last_frame_loss,
                use_reg_loss=opt.use_reg_loss,
                dim_ws=dim_ws,
                use_meanflow=self.use_meanflow,
                meanflow_mode=self.meanflow_mode,
                use_pose_branch=opt.use_pose_branch,   # pose branch: enable lightweight pose residual adapter.
                pose_branch_hidden_dim=opt.pose_branch_hidden_dim,
                pose_branch_dropout=opt.pose_branch_dropout,
                pose_branch_residual_scale=opt.pose_branch_residual_scale,
                pose_branch_gate_bias=opt.pose_branch_gate_bias,
                lambda_pva=opt.lambda_pva,   # dyn loss: expose MeanFlow P/V/A weight as an option.
                dyn_lambda_var=opt.dyn_lambda_var,   # dyn loss: shared temporal variance auxiliary loss.
                dyn_lambda_spec=opt.dyn_lambda_spec,   # dyn loss: shared temporal spectrum auxiliary loss.
                dyn_lambda_diff=opt.dyn_lambda_diff,   # dyn loss: shared temporal diff auxiliary loss.
                dyn_var_use_log=opt.dyn_var_use_log,
                dyn_spec_use_log_power=opt.dyn_spec_use_log_power,
                dyn_spec_highfreq_ratio=opt.dyn_spec_highfreq_ratio,
                dyn_spec_highfreq_weight=opt.dyn_spec_highfreq_weight,
                dyn_diff_velocity_weight=opt.dyn_diff_velocity_weight,
                dyn_diff_acceleration_weight=opt.dyn_diff_acceleration_weight,
                dyn_loss_type=opt.dyn_loss_type,
                dyn_schedule_start_ratio=opt.dyn_schedule_start_ratio,
                dyn_schedule_end_ratio=opt.dyn_schedule_end_ratio,
                dyn_schedule_max_value=opt.dyn_schedule_max_value,
                dyn_active_weighting=opt.dyn_active_weighting,
                dyn_active_weight_mode=opt.dyn_active_weight_mode,
                dyn_active_weight_power=opt.dyn_active_weight_power,
                dyn_active_weight_min=opt.dyn_active_weight_min,
                dyn_active_weight_max=opt.dyn_active_weight_max,
                dyn_active_weight_normalize=opt.dyn_active_weight_normalize,
                dyn_active_weight_detach=opt.dyn_active_weight_detach,
                dyn_loss_parts=opt.dyn_loss_parts,   # dyn loss: select which motion latent parts receive L_var/L_spec/L_diff
            )

        return lmdm

    def _configure_pose_branch_training(self):
        opt = self.opt
        if not getattr(opt, "freeze_backbone_for_pose_branch", False):
            return
        if not getattr(opt, "use_pose_branch", False):
            raise ValueError("freeze_backbone_for_pose_branch=True requires use_pose_branch=True")

        trainable_count = 0
        frozen_count = 0
        for name, param in self.LMDM.model.named_parameters():
            is_pose_branch = "pose_branch" in name
            param.requires_grad = is_pose_branch
            if is_pose_branch:
                trainable_count += param.numel()
            else:
                frozen_count += param.numel()

        if trainable_count == 0:
            raise ValueError("No pose_branch parameters found to train.")

        if self.is_main_process:
            print(
                "[POSE BRANCH] Frozen backbone parameters: "
                f"{frozen_count:,}; trainable pose_branch parameters: {trainable_count:,}"
            )

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
        trainable_params = [(name, param) for name, param in self.LMDM.model.named_parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found for optimizer.")

        pose_params = [param for name, param in trainable_params if "pose_branch" in name]
        backbone_params = [param for name, param in trainable_params if "pose_branch" not in name]

        if getattr(opt, "freeze_backbone_for_pose_branch", False):
            optimizer_params = [{"params": pose_params, "lr": opt.pose_branch_lr}]
            print(f"[POSE BRANCH] Optimizer trains pose_branch only with lr={opt.pose_branch_lr}")
        elif pose_params and getattr(opt, "pose_branch_lr", opt.lr) != opt.lr:
            optimizer_params = [
                {"params": backbone_params, "lr": opt.lr},
                {"params": pose_params, "lr": opt.pose_branch_lr},
            ]
            print(f"[POSE BRANCH] Optimizer lr groups: backbone={opt.lr}, pose_branch={opt.pose_branch_lr}")
        else:
            optimizer_params = [param for _, param in trainable_params]
        
        # Optimizer 선택 (MeanFlow 논문 권장: Adam with lr=1e-4)
        optimizer_type = getattr(opt, 'optimizer', 'adan').lower()
        if optimizer_type == "adam":
            import torch.optim as optim_module
            optim = optim_module.Adam(optimizer_params, lr=opt.lr, weight_decay=0.02)
            print(f"[OPTIMIZER] Using Adam with lr={opt.lr}")
        else:
            optim = Adan(optimizer_params, lr=opt.lr, weight_decay=0.02)
            print(f"[OPTIMIZER] Using Adan with lr={opt.lr}")

        ##### Load optimizer state from checkpoint if available 체크포인트 로딩 (_init_optim)
        if opt.checkpoint and os.path.exists(opt.checkpoint):
            try:
                ckpt = torch.load(opt.checkpoint, map_location='cpu')
                if 'optimizer_state_dict' in ckpt:
                    ckpt_groups = ckpt['optimizer_state_dict'].get('param_groups', [])
                    cur_groups = optim.state_dict().get('param_groups', [])
                    groups_match = (
                        len(ckpt_groups) == len(cur_groups)
                        and all(len(a.get('params', [])) == len(b.get('params', [])) for a, b in zip(ckpt_groups, cur_groups))
                    )
                    if groups_match:
                        optim.load_state_dict(ckpt['optimizer_state_dict'])
                        print(f"[RESUME] Optimizer state loaded from {opt.checkpoint}")
                    else:
                        print("[RESUME] Optimizer state skipped because trainable parameter groups changed")
                else:
                    print(f"[WARNING] No optimizer state found in checkpoint: {opt.checkpoint}")
            except Exception as e:
                print(f"[WARNING] Failed to load optimizer state: {e}")

        return optim

    def _init_log(self):
        opt = self.opt

        # Add timestamp to experiment name (Korean timezone)
        KST = timezone(timedelta(hours=9))
        # timestamp = datetime.datetime.now(KST).strftime("%y%m%d_%H%M")
        timestamp = datetime.datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        experiment_name_with_time = f"{opt.experiment_name}_{timestamp}"
        # epoch sample: keep filesystem experiment names stable while allowing explicit wandb names.
        wandb_run_name = opt.wandb_run_name.strip() if getattr(opt, "wandb_run_name", "") else experiment_name_with_time

        experiment_path = os.path.join(opt.experiment_dir, experiment_name_with_time)
        self.error_log_path = os.path.join(experiment_path, 'error')

        if not self.is_main_process:
            return

        # ckpt
        self.ckpt_path = os.path.join(experiment_path, 'weights')   # 100 에폭마다 가중치 저장
        os.makedirs(self.ckpt_path, exist_ok=True)

        # save opt
        opt_pkl = os.path.join(experiment_path, 'opt.pkl')          # 옵션 설정 저장
        dump_pkl(vars(opt), opt_pkl)

        # loss log
        loss_log = os.path.join(experiment_path, 'loss.log')        # 학습 손실 로그 저장
        self.loss_logger = open(loss_log, 'a')

        self.ckpt_file_list_for_clear = []

        ##### sample generation
        self.sample_dir = os.path.join(experiment_path, 'samples') if self.is_main_process else None    # 100 에폭마다 샘플 영상 저장
        if self.is_main_process:
            os.makedirs(self.sample_dir, exist_ok=True)

            # Initialize wandb for experiment tracking
            if WANDB_AVAILABLE and wandb is not None:
                print(f"[WANDB] Initializing wandb - project: {opt.wandb_pj_name}, name: {wandb_run_name}")
                print(f"[WANDB] wandb_log_freq: {opt.wandb_log_freq} iterations")
                wandb.init(project=opt.wandb_pj_name, name=wandb_run_name)
                print(f"[WANDB] Initialized successfully! Run URL: {wandb.run.url if wandb.run else 'N/A'}")

    def _loss_backward(self, loss):
        self.optim.zero_grad()

        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        self.optim.step()

    def _train_one_step(self, data_dict):
        x = data_dict["kp_seq"]             # (B, L, kp_dim)- Motion latent
        cond_frame = data_dict["kp_cond"]   # (B, kp_dim)- Condition frame
        cond = data_dict["aud_cond"]        # (B, L, aud_dim)- Audio + Conditional features

        if not self.opt.use_accelerate:
            x = x.to(self.device)
            cond_frame = cond_frame.to(self.device)
            cond = cond.to(self.device)
            # dyn loss: forward step information so both original Ditto and MeanFlow
            # can optionally use schedule-aware auxiliary losses later.
            loss, loss_dict = self.LMDM.diffusion(
                x,
                cond_frame,
                cond,
                global_step=self.global_step,
                total_steps=self.total_steps,
            )
        else:
            with self.accelerator.autocast():
                # dyn loss: same scheduled-loss interface under autocast.
                loss, loss_dict = self.LMDM.diffusion(
                    x,
                    cond_frame,
                    cond,
                    global_step=self.global_step,
                    total_steps=self.total_steps,
                )

        # Log to wandb at iteration frequency
        if self.is_main_process and WANDB_AVAILABLE and wandb is not None and self.global_step % self.opt.wandb_log_freq == 0:
            log_dict = {
                "Global_Step": self.global_step,
                "Epoch": self.epoch,
                "total_loss": float(loss),
                **{k: float(v) for k, v in loss_dict.items()}
            }
            wandb.log(log_dict, step=self.global_step)
            # Debug: confirm wandb logging
            if self.global_step <= 100:  # Only print for first few logs
                print(f"[WANDB] Logged at step {self.global_step}: loss={float(loss):.4f}")

        return loss, loss_dict

    def _train_one_epoch(self):
        data_loader = self.data_loader

        DAM = DictAverageMeter()

        self.LMDM.train()
        self.local_step = 0
        pbar = tqdm(data_loader, disable=not self.is_main_process, desc=f"Epoch {self.epoch}")
        for data_dict in pbar:
            self.global_step += 1
            self.local_step += 1

            loss, loss_dict = self._train_one_step(data_dict)
            self._loss_backward(loss)

            if self.is_main_process:
                loss_dict['total_loss'] = loss
                loss_dict_val = {k: float(v) for k, v in loss_dict.items()}
                DAM.update(loss_dict_val)
                
                ##### Update progress bar with current loss
                avg_loss = DAM.average()
                loss_str = f"loss: {avg_loss.get('total_loss', 0):.4f}"
                # dyn loss: surface auxiliary totals in the progress bar for quick checks.
                if 'dyn_total' in avg_loss:
                    loss_str += f" | dyn: {avg_loss['dyn_total']:.4f}"
                if 'pva_total' in avg_loss:
                    loss_str += f" | pva: {avg_loss['pva_total']:.4f}"
                if 'pva_loss' in avg_loss:
                    loss_str += f" | pva: {avg_loss['pva_loss']:.4f}"
                if 'last_frame_loss' in avg_loss:
                    loss_str += f" | lf: {avg_loss['last_frame_loss']:.4f}"
                pbar.set_postfix_str(loss_str)

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

        # save model only if epoch % save_ckpt_freq == 0
        if epoch % self.opt.save_ckpt_freq == 0:
            if self.accelerator is not None:
                state_dict = self.accelerator.unwrap_model(self.LMDM.model).state_dict()
            else:
                state_dict = self.LMDM.model.state_dict()

            ckpt = {
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optim.state_dict(), # 옵티마이저 상태 저장
                "epoch": epoch, # 에폭 정보 저장
                "global_step": self.global_step,     # 글로벌 스텝 저장               
            }
            ckpt_p = os.path.join(self.ckpt_path, f"train_{epoch}.pt")
            torch.save(ckpt, ckpt_p)
            tqdm.write(f"[MODEL SAVED at Epoch {epoch}]")

            # add to clear list for cleanup
            # self.ckpt_file_list_for_clear.append(ckpt_p)

            # # clear old models (keep last 5 checkpoints)
            # if len(self.ckpt_file_list_for_clear) > 5:
            #     _ckpt = self.ckpt_file_list_for_clear.pop(0)
            #     try:
            #         os.remove(_ckpt)
            #         tqdm.write(f"[OLD CHECKPOINT REMOVED] {_ckpt}")
            #     except:
            #         traceback.print_exc()
            #         self.ckpt_file_list_for_clear.insert(0, _ckpt)

            # Log 'epoch-averaged losses' to wandb if available
            if WANDB_AVAILABLE:
                log_dict = {
                    "Epoch": epoch,
                    # **DAM.average()
                    **{f"epoch_avg_{k}": v for k, v in DAM.average().items()}
                }
                wandb.log(log_dict)
            else:
                print(f"Epoch {epoch}: Loss = {DAM.average().get('total_loss', 'N/A'):.6f}")

        ##### epoch sample: generate one sample video per epoch, or by interval if disabled.
        if self._should_generate_sample(epoch):
            try:
                self.generate_sample_video(self.opt, epoch)
            except Exception as e:
                print(f"[WARNING] Failed to generate sample video at epoch {epoch}: {e}")
                traceback.print_exc()

        # clear model
        # if epoch % self.opt.save_ckpt_freq != 0:
        #     self.ckpt_file_list_for_clear.append(ckpt_p)
        
        # if len(self.ckpt_file_list_for_clear) > 5:
        #     _ckpt = self.ckpt_file_list_for_clear.pop(0)
        #     try:
        #         os.remove(_ckpt)
        #     except:
        #         traceback.print_exc()
        #         self.ckpt_file_list_for_clear.insert(0, _ckpt)

    def _train_loop(self):
        print(time.asctime(), 'start ...')

        opt = self.opt

        ##### Determine start epoch and global_step based on checkpoint 학습 재개 로직 (_train_loop)
        start_epoch = 1
        self.global_step = 0
        ##### ?? 에촉부터 시작 
        if opt.checkpoint and os.path.exists(opt.checkpoint):
            try:
                ckpt = torch.load(opt.checkpoint, map_location='cpu')
                if 'epoch' in ckpt:
                    checkpoint_epoch = ckpt['epoch']
                    start_epoch = checkpoint_epoch + 1
                else:
                    # Fallback: Extract epoch from checkpoint filename
                    ckpt_filename = os.path.basename(opt.checkpoint)
                    if ckpt_filename.startswith('train_') and ckpt_filename.endswith('.pt'):
                        checkpoint_epoch = int(ckpt_filename.split('_')[1].split('.')[0])
                        start_epoch = checkpoint_epoch + 1

                if 'global_step' in ckpt:
                    self.global_step = ckpt['global_step']

                print(f"[RESUME] Resuming from epoch {checkpoint_epoch}, global_step {self.global_step}, starting from epoch {start_epoch}")
            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint info: {e}")
                # Fallback to filename parsing
                ckpt_filename = os.path.basename(opt.checkpoint)
                if ckpt_filename.startswith('train_') and ckpt_filename.endswith('.pt'):
                    try:
                        checkpoint_epoch = int(ckpt_filename.split('_')[1].split('.')[0])
                        start_epoch = checkpoint_epoch + 1
                        print(f"[RESUME] Resuming from epoch {checkpoint_epoch}, starting from epoch {start_epoch} (filename fallback)")
                    except (ValueError, IndexError):    
                        print(f"[WARNING] Could not parse epoch from checkpoint filename: {ckpt_filename}")
        #####
        self.local_step = 0
        epoch_pbar = trange(start_epoch, opt.epochs + 1, disable=not self.is_main_process, desc="Training")
        for epoch in epoch_pbar:
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            self.epoch = epoch
            DAM = self._train_one_epoch()

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            if self.is_main_process:
                self.LMDM.eval()
                ##### Update epoch progress bar with average loss
                avg_loss = DAM.average()
                epoch_pbar.set_postfix_str(f"avg_loss: {avg_loss.get('total_loss', 0):.4f}")
                self._show_and_save(DAM)

        print(time.asctime(), 'done.')

        if self.is_main_process and WANDB_AVAILABLE:
            wandb.run.finish()

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

    def _should_generate_sample(self, epoch):
        # epoch sample: centralize sample scheduling so every-epoch and interval modes are easy to switch.
        if getattr(self.opt, "sample_every_epoch", False):
            return True
        sample_interval = getattr(self.opt, "sample_interval", 0)
        return sample_interval > 0 and epoch % sample_interval == 0

    ##### Generate sample video for visual evaluation during training
    def generate_sample_video(self, opt, epoch):
        """
        Generate sample video for visual evaluation during training.
        Uses StreamSDK similar to inference.py
        """
        # if not self.accelerator.is_main_process:
        if not self.is_main_process:
            return

        # Check if sample generation is enabled
        if not hasattr(opt, 'sample_audio_path') or not opt.sample_audio_path:
            return
        if not hasattr(opt, 'sample_source_path') or not opt.sample_source_path:
            return
        if not hasattr(opt, 'sample_cfg_pkl') or not opt.sample_cfg_pkl:
            return
        if not hasattr(opt, 'sample_data_root') or not opt.sample_data_root:
            return

        StreamSDKCls = _get_stream_sdk()

        # Check if StreamSDK is available
        if StreamSDKCls is None:
            print("[WARNING] StreamSDK not available - skipping sample video generation")
            return

        # Check if files exist
        if not os.path.exists(opt.sample_audio_path):
            print(f"[WARNING] Sample audio not found: {opt.sample_audio_path}")
            return
        if not os.path.exists(opt.sample_source_path):
            print(f"[WARNING] Sample source not found: {opt.sample_source_path}")
            return
        if not os.path.exists(opt.sample_cfg_pkl):
            print(f"[WARNING] Sample cfg_pkl not found: {opt.sample_cfg_pkl}")
            return

        try:
            # Import torch locally to avoid any shadowing issues
            import torch as _torch
            
            print(f"\n[SAMPLE] Generating sample video at epoch {epoch}...")

            # Save current checkpoint temporarily for SDK to load
            
            # temp_ckpt_path = os.path.join(self.ckpt_path, f"train_{epoch}.pt")
            # if not os.path.exists(temp_ckpt_path):
            #     print(f"[WARNING] Checkpoint not found: {temp_ckpt_path}")
            #     return
            temp_ckpt_path = os.path.join(self.sample_dir, f"temp_checkpoint_epoch_{epoch}.pt")

            # Save current model state for sample generation
            if self.accelerator is not None:
                state_dict = self.accelerator.unwrap_model(self.LMDM.model).state_dict()
            else:
                state_dict = self.LMDM.model.state_dict()

            temp_ckpt = {"model_state_dict": state_dict}
            _torch.save(temp_ckpt, temp_ckpt_path)
            print(f"[SAMPLE] Temporary checkpoint saved: {temp_ckpt_path}")

            # Create sample output directory
            sample_dir = self.sample_dir

            # Output paths
            output_path = os.path.join(sample_dir, f"sample_epoch_{epoch:04d}.mp4")
            tmp_output_path = os.path.join(sample_dir, f"sample_epoch_{epoch:04d}_tmp.mp4")

            # Initialize SDK with current checkpoint
            SDK = StreamSDKCls(
                opt.sample_cfg_pkl,
                opt.sample_data_root,
                checkpoint_path=temp_ckpt_path,
                use_meanflow=self.use_meanflow,  # 학습 모드에 따라 자동 선택
                # pose branch: sample inference must build the same decoder
                # structure as the currently training checkpoint.
                use_pose_branch=opt.use_pose_branch,
                pose_branch_hidden_dim=opt.pose_branch_hidden_dim,
                pose_branch_dropout=opt.pose_branch_dropout,
                pose_branch_residual_scale=opt.pose_branch_residual_scale,
                pose_branch_gate_bias=opt.pose_branch_gate_bias,
            )

            # Setup SDK
            SDK.setup(opt.sample_source_path, tmp_output_path)

            # Load audio and calculate frames
            audio, sr = librosa.core.load(opt.sample_audio_path, sr=16000)
            num_f = math.ceil(len(audio) / 16000 * 25)

            # Setup N_d
            SDK.setup_Nd(N_d=num_f, fade_in=-1, fade_out=-1, ctrl_info={})

            # Run inference (offline mode)
            aud_feat = SDK.wav2feat.wav2feat(audio)
            SDK.audio2motion_queue.put(aud_feat)
            SDK.close()

            # Merge audio with video
            cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{opt.sample_audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
            os.system(cmd)

            # Remove temp files
            if os.path.exists(tmp_output_path):
                os.remove(tmp_output_path)
            if os.path.exists(SDK.tmp_output_path):
                os.remove(SDK.tmp_output_path)
            if os.path.exists(temp_ckpt_path):
                os.remove(temp_ckpt_path)
                print(f"[SAMPLE] Temporary checkpoint removed: {temp_ckpt_path}")

            print(f"[SAMPLE] Sample video saved: {output_path}")

            # Log to wandb if available
            if WANDB_AVAILABLE:
                try:
                    # epoch sample: log each generated epoch video with its epoch number.
                    wandb.log({
                        "sample_video": wandb.Video(output_path, fps=25, format="mp4"),
                        "sample_epoch": epoch,
                    }, step=self.global_step)
                except Exception as e:
                    print(f"[WARNING] Failed to log video to wandb: {e}")

        except Exception as e:
            print(f"[WARNING] Failed to generate sample video: {e}")
            import traceback
            traceback.print_exc()
