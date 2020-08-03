CUDA_VISIBLE_DEVICES=3 python main.py
--texturePath=/home/chenqi/dataset/materials/
--ngf=80 --ndf=80 --zLoc=100 --zGL=31 --nDep=5
--nDepD=5 --batchSize=24 --niter=10000
--coeff_color_loss=0 --imageSize 160
--use_perceptual_loss=1