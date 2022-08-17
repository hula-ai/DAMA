# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------
# Reference https://github.com/facebookresearch/mae

"""
python submitit_pretrain.py --arch main_vit_base \
      --img_size 128 --patch_size 16 --in_chans 7 \
      --batch_size 128 --epochs 500 --warmup_epochs 40 --stable_epoch 0 --blr 1.5e-4 --accum_iter 1 \
      --mask_ratio 0.8 --mask_overlap_ratio 0.5 --last_k_blocks 6 --norm_pix_loss \
      --data_path data_path \
      --job_dir output_dir \
      --code_dir code_base_dir \
      --nodes 1 --ngpus 4
"""

import argparse
import os
import uuid
from pathlib import Path

import main_pretrain as trainer
import submitit


def parse_args():
    trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for DAMA pretrain", parents=[trainer_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--code_dir", default="", type=str, help="Copy code folder to job_dir")

    parser.add_argument("--partition", default="batch", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Request 32G V100 GPUs")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    return parser.parse_args()

def get_shared_folder(tensordir) -> Path:
    # user = os.getenv("USER")
    if Path(tensordir).is_dir():
        p = Path(tensordir)
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file(tensordir):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(tensordir)), exist_ok=True)
    init_file = get_shared_folder(tensordir) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main_pretrain as trainer

        self._setup_gpu_args()
        trainer.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(args.job_dir).as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

def copy_folder(src, dst):
    import shutil
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    if not os.path.exists(args.job_dir):
        os.makedirs(args.job_dir)

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=10 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=2,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="DAMA")

    args.dist_url = get_init_file(args.job_dir).as_uri()
    args.output_dir = args.job_dir

    args.comment = ''


    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)
    print(job.job_id)


    # src_copy = 'DAMA folder/' # create a copy of DAMA code to output folder
    src_copy = args.code_dir
    dst=args.output_dir+'/DAMA'
    copy_folder(src_copy, dst)

if __name__ == "__main__":
    main()
