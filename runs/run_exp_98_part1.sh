export CUDA_VISIBLE_DEVICES=1
cd ..
sleep 3
nohup /home/admin/miniconda3/envs/son.nh/bin/python client.py --config="federated_learning/config/test.json" --client_idx=00 > ./logs_clients/client_00_node8.txt &
sleep 3
nohup /home/admin/miniconda3/envs/son.nh/bin/python client.py --config="federated_learning/config/test.json" --client_idx=01 > ./logs_clients/client_01_node8.txt &
sleep 3
nohup /home/admin/miniconda3/envs/son.nh/bin/python client.py --config="federated_learning/config/test.json" --client_idx=02 > ./logs_clients/client_02_node8.txt &
sleep 3
nohup /home/admin/miniconda3/envs/son.nh/bin/python client.py --config="federated_learning/config/test.json" --client_idx=03 > ./logs_clients/client_03_node8.txt &
sleep 3
nohup /home/admin/miniconda3/envs/son.nh/bin/python client.py --config="federated_learning/config/test.json" --client_idx=04 > ./logs_clients/client_04_node8.txt &
sleep 3
nohup /home/admin/miniconda3/envs/son.nh/bin/python client.py --config="federated_learning/config/test.json" --client_idx=05  &
sleep 3
nohup /home/admin/miniconda3/envs/son.nh/bin/python client.py --config="federated_learning/config/test.json" --client_idx=06  &
sleep 3
nohup /home/admin/miniconda3/envs/son.nh/bin/python client.py --config="federated_learning/config/test.json" --client_idx=07  &
sleep 3
nohup /home/admin/miniconda3/envs/son.nh/bin/python client.py --config="federated_learning/config/test.json" --client_idx=08  &
sleep 3
nohup /home/admin/miniconda3/envs/son.nh/bin/python client.py --config="federated_learning/config/test.json" --client_idx=09  &