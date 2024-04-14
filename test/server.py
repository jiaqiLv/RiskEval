from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

# 定义指标聚合方法
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

# 定义模型聚合策略
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# 启动 Flower 服务端
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)