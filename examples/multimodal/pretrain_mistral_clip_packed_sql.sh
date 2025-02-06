#!/bin/bash
# Pretrain a multimodal model.

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
MODEL_NAME="mcore-llava-mistral-7b-instruct-clip336-pretraining"


WORKSPACE=output
# Check that the user has set an output path for model checkpoints.
if [[ -z $WORKSPACE ]]; then
    echo "Please set WORKSPACE for storing your model checkpoints."
    exit 1
fi

SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

# if [[ -z $LOAD_NAME ]]; then
#     echo "Please set LOAD_NAME for input model name."
#     exit 1
# fi

TOKENIZER_MODEL=/workspace/data/llm/Mistral-7B-v0.3
if [[ -z $TOKENIZER_MODEL ]]; then
    echo "Please set TOKENIZER_MODEL for tokenizer model name."
    exit 1
fi

echo $FINETUNE_DIR

CHECKPOINT_DIR="${WORKSPACE}/${LOAD_NAME}/checkpoints"

DATA_TRAIN="${SOURCE}/examples/multimodal/pretrain_dataset2.yaml"

echo DATA_TRAIN: $DATA_TRAIN

DEBUG=1
if [[ $DEBUG -eq 1 ]]; then
    BZ=1
    NW=2
    HD=0.0
    LI=10
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
else
    BZ=256
    NW=2
    HD=0.1
    LI=10
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
fi


OPTIONS=" \
    --apply-layernorm-1p \
    --attention-softmax-in-fp32 \
    --use-distributed-optimizer \
    --transformer-impl transformer_engine \
    --use-te \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --num-workers ${NW} \
    --exit-duration-in-mins 230 \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout ${HD} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 8192
    --decoder-seq-length 9000 \
    --max-position-embeddings 9000 \
    --ffn-hidden-size 14336 \
    --train-iters 20000 \
    --micro-batch-size 1 \
    --global-batch-size ${BZ} \
    --lr-decay-iters 20000 \
    --lr-warmup-fraction .01 \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 10 \
    --eval-interval 1000 \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tokenizer-prompt-format mistral \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${SOURCE}/examples/multimodal/manual_prompts.json \
    --save-interval 1000 \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --split 100,0,0 \
    --clip-grad 1.0 \
    --weight-decay 1e-2 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --eod-mask-loss \
    --freeze-LM \
    --freeze-ViT \
    --patch-dim 14 \
    --img-h 224 \
    --img-w 224 \
    --dataloader-type external \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --language-model-type=mistral_7b \
    --disable-vision-class-token \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --allow-missing-vision-projection-checkpoint \
    --packing-seq-length 8192 \
    --packing-buffer-size 100 \
    --ckpt-format torch
"

    # --packing-seq-length 8192 \
    # --packing-buffer-size 500 \
    # --packing-seq-length 4096 \
    # --packing-buffer-size 100 \

export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

torchrun --nproc_per_node 1 examples/multimodal/train.py ${OPTIONS}