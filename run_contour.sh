mpirun -n 6 python plot_surface.py --mpi --cuda --model resnet18 --x=-10:45:26 --y=-9:12:26 \
--model_file cifar10/trained_nets/resnet18/model_295.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot \
--proj_file cifar10/trained_nets/resnet18/PCA_weights_save_epoch=2/directions.h5_proj_cos.h5 \
--dir_file cifar10/trained_nets/resnet18/PCA_weights_save_epoch=2/directions.h5
