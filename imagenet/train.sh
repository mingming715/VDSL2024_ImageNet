# train.sh
export CUDA_VISIBLE_DEVICES=$1
#python main_adv_aa.py /database/Data/CLS-LOC -a 'efficientnet-b1' --learning-rate 0.1 --resume '/root/LGD2024/examples/imagenet/new_adv_model_best_b1_1718867880.pth.tar' --workers 32 --batch-size 128 --epoch 120 -at

# vanilla train
#python main.py /database/Data/CLS-LOC --arch 'efficientnet-b1' --workers 16 --batch-size 256 --epochs 200 --lr 0.001 --weight-decay 1e-5 --seed 42

# finetune blurpool
python main.py /database/Data/CLS-LOC -a 'efficientnet-b1' --lr 0.0025 --workers 8 --batch-size 1 --epoch 50 --seed 42

# evaluate
#python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --pretrained -bw '/root/LGD2024/examples/imagenet/model_best/model_best_240816_060846.pth.tar'  --lr 0.008 --workers 8 --batch-size 256

#train
# python main.py /database/Data/CLS-LOC -a 'efficientnet-b1' -bp --lr 0.002 --workers 16 --batch-size 256 --resume '/root/LGD2024/examples/imagenet/model_best_b1_1715683639.pth.tar' --epoch 180

#train_ltn
#python main_ltn.py /database/Data/CLS-LOC -a 'efficientnet-b1' --workers 8 --batch-size 4 --epoch 90

#evaluate
#python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --pretrained --gpu 0 --batch-size 512 --advprop

#evaluate (Before BatchNorm)
#python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --pretrained -bw '/root/LGD2024/examples/imagenet/model_best_b1_1714978364.pth.tar' --workers 16 --batch-size 96

#evaluate (After activation)
# python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --pretrained -bw '/root/LGD2024/examples/imagenet/af_ac_epoch119_checkpoint_b1_1714729837.pth.tar' --workers 16 --batch-size 96

#evaluate (Before Activation)
#python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --pretrained -bw '/root/LGD2024/examples/imagenet/model_best_b1_1715683639.pth.tar' --workers 16 --batch-size 256

#evaluate (Pretrained B1) - No BlurPool, 제공된 weight
#python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --pretrained --workers 16 --batch-size 96

#evaluate (Pretrained B1) - No BlurPool, 직접 훈련한 weight
#python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --pretrained -tw '/root/LGD2024/examples/imagenet/model_best/model_best_240706_021821.pth.tar' --workers 16 --batch-size 256

#evaluate (After activation, only MB)
#python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --pretrained -bw '/root/LGD2024/examples/imagenet/model_best_b1_1715683639.pth.tar' --workers 16 --batch-size 256
