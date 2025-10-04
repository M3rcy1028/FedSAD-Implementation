'''
    Declare hyperparameters
'''
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--percentile', type=int, default=84)
    parser.add_argument('--random_seed', type=int, default=123)

    '''
        FL 관점에서 client-server이고, 
        server간 연합학습이므로,    client는 각 server가 되고
                                server는 각 server를 집계하는 center임
    '''
    # clients
    parser.add_argument('--client_nums', type=int, default=15)   # 서버수가 50까지 되려나..? 될수도..?
    parser.add_argument('--client_epochs', type=int, default=20) # == local epochs

    # servers
    # parser.add_argument('--server_nums', default=6)
    parser.add_argument('--set_verbose', type=int, default=2)
    parser.add_argument('--server_rounds', type=int, default=10) # server aggregation


    args, _ = parser.parse_known_args()
    return args