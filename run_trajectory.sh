#python plot_trajectory.py --model resnet18 --max_epoch 296 --model_folder ./cifar10/trained_nets/resnet18  --save_epoch 4
python plot_trajectory.py --model flight_random --dataset flight --max_epoch 400 --prefix epoch_ --suffix .pth  --model_folder ./flight/trained_nets/random  --save_epoch 4
#python plot_trajectory.py --model flight_finetune --dataset flight --max_epoch 400 --prefix epoch_ --suffix .pth --model_folder ./flight/trained_nets/finetune  --save_epoch 4
