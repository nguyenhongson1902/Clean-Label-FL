cd ..
sleep 3
nohup /home/sonnh/miniconda3/envs/pytorch-gpu/bin/python client.py --config="federated_learning/config/test.json" --client_idx=00 > ./logs_clients/client_00.txt &
sleep 3
nohup /home/sonnh/miniconda3/envs/pytorch-gpu/bin/python client.py --config="federated_learning/config/test.json" --client_idx=01 > ./logs_clients/client_01.txt &
sleep 3
nohup /home/sonnh/miniconda3/envs/pytorch-gpu/bin/python client.py --config="federated_learning/config/test.json" --client_idx=02 > ./logs_clients/client_02.txt &
sleep 3
nohup /home/sonnh/miniconda3/envs/pytorch-gpu/bin/python client.py --config="federated_learning/config/test.json" --client_idx=03 > ./logs_clients/client_03.txt &
sleep 3
nohup /home/sonnh/miniconda3/envs/pytorch-gpu/bin/python client.py --config="federated_learning/config/test.json" --client_idx=04 > ./logs_clients/client_04.txt &