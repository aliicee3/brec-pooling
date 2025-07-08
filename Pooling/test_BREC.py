# This program is the pipeline for testing expressiveness.
# It includes 4 stages:
#   1. pre-calculation;
#   2. dataset construction;
#   3. model construction;
#   4. evaluation

import numpy as np
import torch
import torch_geometric
import torch_geometric.loader
from loguru import logger
import time
from BRECDataset_v3 import BRECDataset
# from RIGIDDataset import RIGIDDataset
from tqdm import tqdm
import os
from torch.nn import CosineEmbeddingLoss
import argparse
from pooling.model import GINandPool

parser = argparse.ArgumentParser(description="BREC Test")

parser.add_argument("--EPSILON_MATRIX", type=float, default=1e-7)
parser.add_argument("--EPSILON_CMP", type=float, default=1e-6)
parser.add_argument("--P_NORM", type=str, default="2", choices=['2', 'inf'])
parser.add_argument("--EPOCH", type=int, default=20)
parser.add_argument("--LEARNING_RATE", type=float, default=1e-4)
parser.add_argument("--BATCH_SIZE", type=int, default=16)
parser.add_argument("--WEIGHT_DECAY", type=float, default=0)  # 1e-4
parser.add_argument("--OUTPUT_DIM", type=int, default=16)
parser.add_argument("--NUM_BLOCKS", type=int, default=4)
parser.add_argument("--NUM_LAYERS", type=int, default=4)
parser.add_argument("--SEED", type=int, default=2023)
parser.add_argument("--THRESHOLD", type=float, default=72.34)
parser.add_argument("--MARGIN", type=float, default=0.0)
parser.add_argument("--LOSS_THRESHOLD", type=float, default=0.2)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--POOLING", type=str, default="edge_pool", choices=GINandPool.POOLING_OPTIONS)  # edge_pool
parser.add_argument("--CONV_TYPE", type=str, default="gin")
parser.add_argument("--HIDDEN_DIM", type=int, default=16)
parser.add_argument("--DATASET", type=str, default='BREC_v3', choices=['BREC_v3', 'RIGID'])
parser.add_argument("--ALPHA", type=float, default=0.9999)
parser.add_argument("--MERGE", type=str, default='single', choices=['max', 'single', 'combine'])
parser.add_argument("--PATH", type=str, default=None)
# parser.add_argument("--DATASET", type=str, default='RIGID', choices=['BREC_v3', 'RIGID'])

# General settings.
args = parser.parse_args()

args.P_NORM = 2 if args.P_NORM == "2" else torch.inf
# torch.use_deterministic_algorithms(True)

if args.DATASET == 'RIGID':
    # number of pairs in the dataset
    SAMPLE_NUM = 1086

    # number of vertex-permuted repetitions of pairs
    # if this number is 1, we need to use the hacked T2 test as supposed by the BREC authors
    NUM_RELABEL = 6

    # part_dict: {graph generation type, range} ... note that indices have to be multiplied by two as two consecutive graphs form a pair
    part_dict = RIGIDDataset.part_dict

    dataloader = RIGIDDataset

if args.DATASET == 'BREC_v3':
    # number of pairs in the dataset
    SAMPLE_NUM = 400

    # number of vertex-permuted repetitions of pairs
    NUM_RELABEL = 32

    # part_dict: {graph generation type, range} ... note that indices have to be multiplied by two as two consecutive graphs form a pair
    part_dict = BRECDataset().part_dict

    dataloader = BRECDataset


# Stage 1: pre calculation
# Here is for some calculation without data. e.g. generating all the k-substructures
def pre_calculation(*args, **kwargs):
    time_start = time.process_time()

    # Do something
    ## we don't do anything to the data

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"pre-calculation time cost: {time_cost}")


# Stage 2: dataset construction
# Here is for dataset construction, including data processing
def get_dataset(name, device):
    time_start = time.process_time()

    # Do something
    dataset = dataloader(name=name)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model(args, device, dataset):
    time_start = time.process_time()

    in_channels = 1
    hidden_dim = args.HIDDEN_DIM
    out_channels = args.OUTPUT_DIM

    # Do something
    model = GINandPool(in_channels=in_channels, hidden_dim=hidden_dim, out_channels=out_channels,
                       num_layers=args.NUM_LAYERS, num_blocks=args.NUM_BLOCKS, pool=args.POOLING,
                       conv_type=args.CONV_TYPE, alpha=args.ALPHA, merge=args.MERGE).to(device)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model


import signal
import functools


def time_limited(model, data):
    output = model(data)
    return output


# Stage 4: evaluation
# Here is for evaluation.
def evaluation(dataset, model, path, device, args):
    '''
        When testing on BREC, even on the same graph, the output embedding may be different,
        because numerical precision problem occur on large graphs, and even the same graph is permuted.
        However, if you want to test on some simple graphs without permutation outputting the exact same embedding,
        some modification is needed to avoid computing the inverse matrix of a zero matrix.
    '''

    # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use S_epsilon.
    # S_epsilon = torch.diag(
    #     torch.full(size=(OUTPUT_DIM, 1), fill_value=EPSILON_MATRIX).reshape(-1)
    # ).to(device)
    def T2_calculation(dataset, log_flag=False):
        with torch.no_grad():
            loader = torch_geometric.loader.DataLoader(dataset, batch_size=args.BATCH_SIZE)
            pred_0_list = []
            pred_1_list = []
            for data in loader:
                pred = model(data.to(device)).detach()
                pred_0_list.extend(pred[0::2])
                pred_1_list.extend(pred[1::2])
            X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
            Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
            if log_flag:
                logger.info(f"X_mean = {torch.mean(X, dim=1)}")
                logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
            D = X - Y
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            S = torch.cov(D)
            try:
                inv_S = torch.linalg.pinv(S)
            except torch._C._LinAlgError:
                return torch.tensor(0)
            # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use inv_S with S_epsilon.
            # inv_S = torch.linalg.pinv(S + S_epsilon)
            return torch.mm(torch.mm(D_mean.T, inv_S), D_mean)

    time_start = time.process_time()

    logger.info(args)

    # Do something
    cnt = 0
    truly_identified = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = CosineEmbeddingLoss(margin=args.MARGIN)

    part_result = dict()
    true_part_result = dict()

    for part_name, part_range in part_dict.items():
        logger.info(f"{part_name} part starting ---")
        start = time.time()
        cnt_part = 0
        fail_in_reliability_part = 0

        truly_identified_part = 0

        for id in tqdm(range(part_range[0], part_range[1])):
            logger.info(f"ID: {id}")
            model = get_model(args, device, dataset)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            if args.DATASET == 'BREC_v3':
                v1 = id * NUM_RELABEL * 2
                v2 = (id + 1) * NUM_RELABEL * 2
                dataset_traintest = dataset[
                                    v1: v2
                                    ]
                v3 = (id + SAMPLE_NUM) * NUM_RELABEL * 2
                v4 = (id + SAMPLE_NUM + 1) * NUM_RELABEL * 2
                dataset_reliability = dataset[
                                      v3: v4
                                      ]
            elif args.DATASET == 'RIGID':
                # we just duplicate the same graph multiple times
                dataset_traintest = dataset[
                    id * 2 * 6, id * 2 * 6 + 6, id * 2 * 6 + 1, id * 2 * 6 + 7
                ]
                dataset_reliability = dataset[
                    id * 2 * 6 + 2, id * 2 * 6 + 3, id * 2 * 6 + 4, id * 2 * 6 + 5
                ]
            else:
                raise ValueError(f'unknown dataset argument: {args.DATASET}')
            if (args.DATASET == 'RIGID') and (dataset_reliability[0].num_nodes > 2000):
                continue

            model.train()
            failed = False
            for _ in range(args.EPOCH):
                traintest_loader = torch_geometric.loader.DataLoader(
                    dataset_traintest, batch_size=args.BATCH_SIZE
                )
                loss_all = 0
                for data in traintest_loader:
                    optimizer.zero_grad()
                    pred = time_limited(model, data.to(device))
                    if pred is None:
                        failed = True
                        break
                    p1 = pred[0::2]
                    p2 = pred[1::2]
                    loss = loss_func(
                        pred[0::2],
                        pred[1::2],
                        torch.tensor([-1] * (len(pred) // 2)).to(device),
                    )
                    loss.backward()
                    optimizer.step()
                    loss_all += len(pred) / 2 * loss.item()
                if failed:
                    T_square_traintest, T_square_reliability = 0, 0
                    break
                loss_all /= NUM_RELABEL
                logger.info(f"Loss: {loss_all}")
                if loss_all < args.LOSS_THRESHOLD:
                    logger.info("Early Stop Here")
                    break
                scheduler.step(loss_all)

            model.eval()
            isomorphic_flag = False
            reliability_flag = False
            if not failed:
                T_square_traintest = T2_calculation(dataset_traintest, True)
                T_square_reliability = T2_calculation(dataset_reliability, True)

                if T_square_traintest > args.THRESHOLD and not torch.isclose(
                        T_square_traintest, T_square_reliability, atol=args.EPSILON_CMP
                ):
                    isomorphic_flag = True
                if T_square_reliability < args.THRESHOLD:
                    reliability_flag = True

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id)
                logger.info(f"Correct num in current part: {cnt_part}")
            if not reliability_flag:
                fail_in_reliability += 1
                fail_in_reliability_part += 1
            logger.info(f"isomorphic: {isomorphic_flag} {T_square_traintest}")
            logger.info(f"reliability: {reliability_flag} {T_square_reliability}")

            if isomorphic_flag and reliability_flag:
                truly_identified_part += 1
                truly_identified += 1

        end = time.time()
        time_cost_part = round(end - start, 2)

        part_result[part_name] = cnt_part
        true_part_result[part_name] = truly_identified_part

        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        logger.info(
            f"Fail in reliability: {fail_in_reliability_part} / {part_range[1] - part_range[0]}"
        )

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"evaluation time cost: {time_cost}")

    Acc = round(cnt / SAMPLE_NUM, 2)
    logger.info(f"Correct in {cnt} / {SAMPLE_NUM}, Acc = {Acc}")

    logger.info(f"Fail in reliability: {fail_in_reliability} / {SAMPLE_NUM}")
    logger.info(correct_list)

    logger.add(f"{path}/table.tex", format="{message}", encoding="utf-8")
    logger.info(args)
    logger.info('\n\\begin{tabular}{lll}\n\\toprule\n')
    logger.info(f'\\multirow[c]{{ {len(part_result) + 1} }}{{*}}{{ {args.POOLING} }}')
    for part, cnt_part in part_result.items():
        logger.info(f'& {part} & {true_part_result[part]} / {part_dict[part][1] - part_dict[part][0]} \\\\')
    logger.info(f'& Total & {truly_identified} / {SAMPLE_NUM} \\\\')
    logger.info('\\midrule')

    logger.info('\n\\bottomrule\n\\end{tabular}\n')

    logger.add(f"{path}/result_show.txt", format="{message}", encoding="utf-8")
    logger.info(
        "Real_correct\tCorrect\tFail\tOUTPUT_DIM\tBATCH_SIZE\tLEARNING_RATE\tWEIGHT_DECAY\tSEED"
    )
    logger.info(
        f"{cnt - fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{args.OUTPUT_DIM}\t{args.BATCH_SIZE}\t{args.LEARNING_RATE}\t{args.WEIGHT_DECAY}\t{args.SEED}"
    )
    regular_score = true_part_result['Regular'] + true_part_result['Distance_Regular'] + true_part_result[
        '4-Vertex_Condition']

    return cnt - fail_in_reliability, true_part_result['Basic'], regular_score, true_part_result['Extension'], \
    true_part_result['CFI']


def main():
    scores, basic_scores, regular_scores, ext_scores, cfi_scores = [], [], [], [], []
    for i in range(10):
        start = time.time()
        args.SEED = i
        torch_geometric.seed_everything(i)
        torch.backends.cudnn.deterministic = True

        device = torch.device(args.device)
        if args.PATH is None:
            OUT_PATH = f"result_{args.DATASET}_{args.POOLING}"
        else:
            OUT_PATH = args.PATH
        NAME = args.CONV_TYPE
        path = os.path.join(OUT_PATH, NAME)
        os.makedirs(path, exist_ok=True)

        logger.remove(handler_id=None)
        LOG_NAME = os.path.join(path, "log.txt")
        logger.add(LOG_NAME, rotation="5MB")
        logger.info(args)

        dataset_name = args.DATASET

        pre_calculation()
        dataset = get_dataset(name=dataset_name, device=device)
        model = get_model(args, device, dataset)
        score, basic_score, regular_score, ext_score, cfi_score = evaluation(dataset, model, OUT_PATH, device, args)
        scores.append(score)
        basic_scores.append(basic_score)
        regular_scores.append(regular_score)
        ext_scores.append(ext_score)
        cfi_scores.append(cfi_score)
        # Log average scores and standard deviations
        logger.add(f"{path}/final.txt", format="{message}", encoding="utf-8")
        loggier.info(f'Time for run {i}: {(time.time()-start)%60}')
    logger.info(
        f'Final max results: {np.max(basic_scores)+np.max(regular_scores)+np.max(ext_scores)+np.max(cfi_scores)}, {np.max(basic_scores)}, {np.max(regular_scores)}, {np.max(ext_scores)}, {np.max(cfi_scores)}')


if __name__ == "__main__":
    main()
