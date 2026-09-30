python framework/cifar_minimal_dnn.py --load true --epochs 500 --kappa_sweep 1,2,3,5,10,20,33,50 (--mse)

python framework/cifar_minimal_dnn.py --load true --channel_type geometric_rayleigh --snr 60 --epochs 500 --n_m_sweep 8,16,32,64,128,256,512,1024 --mse