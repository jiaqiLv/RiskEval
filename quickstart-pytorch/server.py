import io
import matplotlib.pyplot as plt
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

def plot_data(data, x_label='X', y_label='Y', title='Plot', file_name="plot.png"):
    # 分解为两个列表
    x_values = [i[0] for i in data]
    y_values = [i[1] for i in data]

    # 使用matplotlib进行作图
    plt.figure()
    plt.plot(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.show()
    plt.savefig(f"imgs/fedprox/{file_name}")

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
strategy = fl.server.strategy.FedProx(evaluate_metrics_aggregation_fn=weighted_average, proximal_mu=0.001)

# Start Flower server
hist = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=30),
    strategy=strategy,
)

prefix = "mobinet-fedprox-2-30-lr_0.001"
print("-"*50)
plot_data(hist.losses_distributed, file_name=f"{prefix}_loss.png")
plot_data(hist.metrics_distributed.get('accuracy'), file_name=f"{prefix}_acc.png")

with open('test_record.txt', 'a+') as f:
    f.write(prefix + '\n')
    f.write(str(hist.losses_distributed) + '\n')
    f.write(str(hist.metrics_distributed.get('accuracy')) + '\n')
    f.write()