export CUDA_VISIBLE_DEVICES=$1
# nohup python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' -n /root/LGD2024/examples/imagenet/noisy_student_efficientnet-b1.pth --pretrained --workers 16 --batch-size 128 > eval_noisy_student.out &
# python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --pretrained -n /root/LGD2024/examples/imagenet/efficientnet-b1.pth --workers 8 --batch-size 256
python main.py /database/Data/CLS-LOC -e -a 'efficientnet-b1' --pretrained -n /root/LGD2024/examples/imagenet/noisy_student_efficientnet-b1.pth --workers 8 --batch-size 256
