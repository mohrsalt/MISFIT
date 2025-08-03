

GPU=0                   # gpu to use
SEED=42;                  # randomness seed for sampling
CHANNELS=64;              # number of model base channels (we use 64 for all experiments)

DATASET='brats';          # brats
MODEL='unet';             # 'unet'
CONTR='t1n'               # contrast to be generate by the network ('t1n', t1c', 't2w', 't2f') - just relevant during training

# settings for sampling/inference
ITERATIONS=1200;          # training iteration (as a multiple of 1k) checkpoint to use for sampling
SAMPLING_STEPS=0;         # number of steps for accelerated sampling, 0 for the default 1000
RUN_DIR="";               # tensorboard dir to be set for the evaluation (displayed at start of training)

# detailed settings (no need to change for reproducing)

CHANNEL_MULT=1,2,2,4,4;
ADDITIVE_SKIP=False;      # Set True to save memory
BATCH_SIZE=1;
IMAGE_SIZE=224;
IN_CHANNELS=48;           # Change to work with different number of conditioning images 8 + 8x (with x number of conditioning images)
NOISE_SCHED='linear';




BATCH_SIZE=1;

DATA_DIR=/home/users/ntu/mohor001/scratch/Task8DataBrats/pseudo_val_set;


COMMON="
--dataset=${DATASET} checked
--num_channels=${CHANNELS} checked 
--class_cond=False checked
--num_res_blocks=2 # checked
--num_heads=1# checked
--learn_sigma=False# checked
--use_scale_shift_norm=False# checked
--attention_resolutions=# checked
--channel_mult=${CHANNEL_MULT}# checked
--diffusion_steps=1000# checked
--noise_schedule=${NOISE_SCHED}# checked
--rescale_learned_sigmas=False# checked
--rescale_timesteps=False# checked
--dims=3# checked
--batch_size=${BATCH_SIZE}# checked
--num_groups=32# checked
--in_channels=${IN_CHANNELS}# checked
--out_channels=8# checked
--bottleneck_attention=False# checked
--resample_2d=False# checked
--renormalize=True# checked
--additive_skips=${ADDITIVE_SKIP}# checked
--use_freq=False# checked
--predict_xstart=True# checked
--contr=${CONTR}# checked
"


SAMPLE="
--data_dir=${DATA_DIR}# checked
--data_mode=${DATA_MODE}
--seed=${SEED}# checked
--image_size=${IMAGE_SIZE}#checked
--use_fp16=False# checked
--model_path=${RUN_DIR}/checkpoints/${DATASET}_${ITERATIONS}000.pt#checked
--devices=${GPU}#checked
--output_dir=./results/${DATASET}_${MODEL}_${ITERATIONS}000/#checked
--num_samples=1000#checked
--use_ddim=False#checked
--sampling_steps=${SAMPLING_STEPS}#checked
--clip_denoised=True# checked
"

python Validation/sample_auto_net.py $SAMPLE $COMMON;
