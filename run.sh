
export CUDA_VISIBLE_DEVICES="0"
python train.py --env HalfCheetah-v1 --seed 1 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env HalfCheetah-v1 --seed 2 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env HalfCheetah-v1 --seed 3 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env HalfCheetah-v1 --seed 4 --kappa 5 --epoch 10000 --alpha 2 &

export CUDA_VISIBLE_DEVICES="1"
python train.py --env Hopper-v1 --seed 1 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env Hopper-v1 --seed 2 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env Hopper-v1 --seed 3 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env Hopper-v1 --seed 4 --kappa 5 --epoch 10000 --alpha 2 &

export CUDA_VISIBLE_DEVICES="2"
python train.py --env Walker2d-v1 --seed 1 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env Walker2d-v1 --seed 2 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env Walker2d-v1 --seed 3 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env Walker2d-v1 --seed 4 --kappa 5 --epoch 10000 --alpha 2 &

export CUDA_VISIBLE_DEVICES="3"
python train.py --env Swimmer-v1 --seed 1 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env Swimmer-v1 --seed 2 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env Swimmer-v1 --seed 3 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env Swimmer-v1 --seed 4 --kappa 5 --epoch 10000 --alpha 2 &

export CUDA_VISIBLE_DEVICES="4"
python train.py --env Humanoid-v1 --seed 1 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env Humanoid-v1 --seed 2 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env Humanoid-v1 --seed 3 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env Humanoid-v1 --seed 4 --kappa 5 --epoch 10000 --alpha 2 &

export CUDA_VISIBLE_DEVICES="5"
python train.py --env BipedalWalkerHardcore-v2 --seed 1 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env BipedalWalkerHardcore-v2 --seed 2 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env BipedalWalkerHardcore-v2 --seed 3 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env BipedalWalkerHardcore-v2 --seed 4 --kappa 5 --epoch 10000 --alpha 2 &

export CUDA_VISIBLE_DEVICES="6"
python train.py --env RoboschoolHumanoid-v1 --seed 1 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env RoboschoolHumanoid-v1 --seed 2 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env RoboschoolHumanoid-v1 --seed 3 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env RoboschoolHumanoid-v1 --seed 4 --kappa 5 --epoch 10000 --alpha 2 &

export CUDA_VISIBLE_DEVICES="7"
python train.py --env RoboschoolHumanoidFlagrun-v1 --seed 1 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env RoboschoolHumanoidFlagrun-v1 --seed 2 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env RoboschoolHumanoidFlagrun-v1 --seed 3 --kappa 5 --epoch 10000 --alpha 2 &
python train.py --env RoboschoolHumanoidFlagrun-v1 --seed 4 --kappa 5 --epoch 10000 --alpha 2 &

wait
echo "done"
