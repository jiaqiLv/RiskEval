# Flower源码阅读

## 1.client部分

### 1.1.定义本地模型

```python
class FlowerClient(fl.client.NumPyClient):
    # 获取本地模型对应的参数
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    # 接收模型参数，并更新本地模型
    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    # 本地模型训练，会先调用 set_parameters() 基于收到的全局模型参数更新本地模型
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    # 基于测试数据集进行测试
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

```

这段代码通过继承`fl.client.NumPyClient`这个类，并重写`get_parameters`、`set_parameters`、`fit`、`evaluate`这四个函数，将自定义的模型`net`(通过`nn.Module`实现)嵌入到联邦学习的框架中（我还看了一下`NumPyClient`这个类，发现这个类是一个抽象类）

### 1.2.启动本地模型

首先我们通过下面的代码启动本地客户端

```python
# 启动 Flower 客户端
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
```

然后我们深入的看一下`start_numpy_client`(`start_client`)的实现（这里只列出了主要代码）

```python
def _start_client_internal(
    *,
    server_address: str,
    load_client_app_fn: Optional[Callable[[], ClientApp]] = None,
    client_fn: Optional[ClientFn] = None,
    client: Optional[Client] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[Union[bytes, str]] = None,
    insecure: Optional[bool] = None,
    transport: Optional[str] = None,
    max_retries: Optional[int] = None,
    max_wait_time: Optional[float] = None,
) -> None:
    # 首先建立与server端的连接
    connection, address, connection_error_type = _init_connection(
        transport, server_address
    )
    
    # 重试配置
    retry_invoker = RetryInvoker(
       ... 
    )

    node_state = NodeState()  # 创建客户端状态对象

    while True:  # 进入无限循环，持续与服务器通信
        sleep_duration: int = 0  # 初始化睡眠时间

        with connection(  # 建立与服务器的连接
            address,
            insecure,
            retry_invoker,
            grpc_max_message_length,
            root_certificates,
        ) as conn:  # 使用 `connection` 上下文管理器管理连接
            receive, send, create_node, delete_node = conn  # 获取连接对象上的方法
            # 返回的conn 是一个上下文管理器，其中实现了send、receive等函数，与服务器进行消息传递；以及 create_node 和 delete_node 来在服务器上创建、删除服务器的临时节点（用于简化客户端节点的管理和消息传递）
            # 注册节点（可选）
            if create_node is not None:
                create_node()  # 调用 `create_node` 方法注册节点

            while True:  # 进入消息处理循环
                # 接收消息
                message = receive()

                if message is None:  # 如果没有收到消息
                    time.sleep(3)  # 等待 3 秒
                    continue  # 继续循环

                # 处理控制消息，通过sleep_duration来控制是否继续训练
                out_message, sleep_duration = handle_control_message(message)

                if out_message:  # 如果有控制消息回复
                    send(out_message)  # 发送回复
                    break  # 退出消息处理循环

                # 注册上下文
                node_state.register_context(run_id=message.metadata.run_id)

                # 检索上下文
                context = node_state.retrieve_context(run_id=message.metadata.run_id)

                # 创建错误回复消息（为了避免 linting 错误，但实际上不会使用）
                reply_message = message.create_error_reply(
                    error=Error(code=ErrorCode.UNKNOWN, reason="Unknown")
                )

                # 加载并执行客户端应用
                try:
                    # 加载客户端应用实例
                    client_app: ClientApp = load_client_app_fn()
                    # 执行客户端应用
                    reply_message = client_app(message=message, context=context)
                except Exception as ex:  # 捕获所有异常
						... # 异常处理
                        # 创建错误回复消息
                        reply_message = message.create_error_reply(
                            error=Error(code=e_code, reason=reason)
                        )
                else:  # 没有异常，更新节点状态
                    node_state.update_context(
                        run_id=message.metadata.run_id, context=context
                    )

                # 发送回复
                send(reply_message)
                log(INFO, "Sent reply")

            # 注销节点（可选）
            if delete_node is not None:
                delete_node()

        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            break

        time.sleep(sleep_duration)
```

### 1.3.本地模型的训练与评估

通过`ClientApp`这个类实现

```python
class ClientApp:
    def __init__(
        self,
        client_fn: Optional[ClientFn] = None,  # Only for backward compatibility
        mods: Optional[List[Mod]] = None,
    ) -> None:
        self._mods: List[Mod] = mods if mods is not None else []

        # Create wrapper function for `handle`
        self._call: Optional[ClientAppCallable] = None
        if client_fn is not None:

            def ffn(
                message: Message,
                context: Context,
            ) -> Message:  # pylint: disable=invalid-name
                out_message = handle_legacy_message_from_msgtype(
                    client_fn=client_fn, message=message, context=context
                )
                return out_message

            # Wrap mods around the wrapped handle function
            self._call = make_ffn(ffn, mods if mods is not None else [])

        # Step functions
        self._train: Optional[ClientAppCallable] = None
        self._evaluate: Optional[ClientAppCallable] = None
        self._query: Optional[ClientAppCallable] = None

    def __call__(self, message: Message, context: Context) -> Message:
        """Execute `ClientApp`."""
        # Execute message using `client_fn`
        if self._call:
            return self._call(message, context)

        # Execute message using a new
        if message.metadata.message_type == MessageType.TRAIN:
            if self._train:
                return self._train(message, context)
            raise ValueError("No `train` function registered")
        if message.metadata.message_type == MessageType.EVALUATE:
            if self._evaluate:
                return self._evaluate(message, context)
            raise ValueError("No `evaluate` function registered")
        if message.metadata.message_type == MessageType.QUERY:
            if self._query:
                return self._query(message, context)
            raise ValueError("No `query` function registered")

        # Message type did not match one of the known message types abvoe
        raise ValueError(f"Unknown message_type: {message.metadata.message_type}")

```

这个类在	`__init__`	中接收了传入的本地模型来初始化，在`__call__`中调用本地模型中的方法（train、evaluate等）

call中调用的是`handle_legacy_message_from_msgtype`这个方法，通过服务器的控制信息来控制调用的模型的方法

```python
def handle_legacy_message_from_msgtype(
    client_fn: ClientFn, message: Message, context: Context
) -> Message:
    """Handle legacy message in the inner most mod."""
    client = client_fn(str(message.metadata.partition_id))

    # Check if NumPyClient is returend
    if isinstance(client, NumPyClient):
        client = client.to_client()
        log(
            WARN,
            "Deprecation Warning: The `client_fn` function must return an instance "
            "of `Client`, but an instance of `NumpyClient` was returned. "
            "Please use `NumPyClient.to_client()` method to convert it to `Client`.",
        )

    client.set_context(context)

    message_type = message.metadata.message_type

    # Handle GetPropertiesIns
    if message_type == MessageTypeLegacy.GET_PROPERTIES:
        get_properties_res = maybe_call_get_properties(
            client=client,
            get_properties_ins=recordset_to_getpropertiesins(message.content),
        )
        out_recordset = getpropertiesres_to_recordset(get_properties_res)
    # Handle GetParametersIns
    elif message_type == MessageTypeLegacy.GET_PARAMETERS:
        get_parameters_res = maybe_call_get_parameters(
            client=client,
            get_parameters_ins=recordset_to_getparametersins(message.content),
        )
        out_recordset = getparametersres_to_recordset(
            get_parameters_res, keep_input=False
        )
    # Handle FitIns
    elif message_type == MessageType.TRAIN:
        fit_res = maybe_call_fit(
            client=client,
            fit_ins=recordset_to_fitins(message.content, keep_input=True),
        )
        out_recordset = fitres_to_recordset(fit_res, keep_input=False)
    # Handle EvaluateIns
    elif message_type == MessageType.EVALUATE:
        evaluate_res = maybe_call_evaluate(
            client=client,
            evaluate_ins=recordset_to_evaluateins(message.content, keep_input=True),
        )
        out_recordset = evaluateres_to_recordset(evaluate_res)
    else:
        raise ValueError(f"Invalid message type: {message_type}")

    # Return Message
    return message.create_reply(out_recordset)
```



## 2.server部分

### 1.优化策略定义

这里是一个简单的FedAvg的实现

```python
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

# 定义模型聚合策略
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
```

### 2.启动服务端

启动时可以传入：服务器端口、训练轮数、优化策略等参数

```python
# 启动 Flower 服务端
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
```

<details> <summary>参数列表</summary> 
    <ul>
        <li><b>server_address</b> : 可选[str]<br>
        &emsp;服务器的IPv4或IPv6地址。默认为"[::]:8080"。</li>

        <li><b>server</b> : 可选[flwr.server.Server] (默认: None)<br>
        &emsp;一个服务器实现，可以是flwr.server.Server或其子类。 如果没有提供实例，则start_server将创建一个。</li>
    
        <li><b>config</b> : 可选[ServerConfig] (默认: None)<br>
        &emsp;当前支持的值有num_rounds (int, 默认: 1) 和 round_timeout（以秒为单位的浮点数，默认: None）。</li>
    
        <li><b>strategy</b> : 可选[flwr.server.Strategy] (默认: None)<br>
        &emsp;flwr.server.strategy.Strategy的抽象基类的实现。 如果没有提供策略，则start_server将使用flwr.server.strategy.FedAvg。</li>
    
        <li><b>client_manager</b> : 可选[flwr.server.ClientManager] (默认: None)<br>
        &emsp;flwr.server.ClientManager的抽象基类的实现。 如果没有提供实现，则start_server将使用 flwr.server.client_manager.SimpleClientManager。</li>
    
        <li><b>grpc_max_message_length</b> : int (默认: 536_870_912, 这等于512MB)<br>
        &emsp;可以与Flower客户端交换的gRPC消息的最大长度。 默认应适合大多数模型。 训练非常大的模型的用户可能需要增加这个值。 请注意，需要以相同的值启动Flower客户端（请参见flwr.client.start_client），否则客户端将不知道增加的限制并阻止更大的消息。</li>
    
        <li><b>certificates</b> : Tuple[bytes, bytes, bytes] (默认: None)<br>
        &emsp;包含根证书，服务器证书和私钥的元组，用于启动安全的SSL启用的服务器</li>
    </ul>

</details>

```python
def start_server(
    *,
    server_address: str = ADDRESS_FLEET_API_GRPC_BIDI,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    certificates: Optional[Tuple[bytes, bytes, bytes]] = None,
) -> History:

    # 构造启动地址
    parsed_address = parse_address(server_address)
    if not parsed_address:
        sys.exit(f"Server IP address ({server_address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # 初始化 Server 对象，此对象中包含实际的模型训练流程的支持
    initialized_server, initialized_config = init_defaults(
        server=server,
        config=config,
        strategy=strategy,
        client_manager=client_manager,
    )

    # 启动 grpc 服务端，用于与客户端进行通信
    grpc_server = start_grpc_server(
        client_manager=initialized_server.client_manager(),
        server_address=address,
        max_message_length=grpc_max_message_length,
        certificates=certificates,
    )

    # 执行训练流程
    hist = run_fl(
        server=initialized_server,
        config=initialized_config,
    )

    # 停止 grpc 服务端
    grpc_server.stop(grace=1)

    return hist
```



### 3.服务端的训练

服务端的训练主要是调用`initialized_server.fit()`方法

```
def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
    # 初始化全局模型参数

    self.parameters = self._get_initial_parameters(timeout=timeout)

    # 执行 num_rounds 轮模型训练
    for current_round in range(1, num_rounds + 1):
        # 执行单轮机器学习模型训练

        res_fit = self.fit_round(
            server_round=current_round,
            timeout=timeout,
        )
        if res_fit is not None:
            parameters_prime, fit_metrics, _ = res_fit
            # 根据聚合生成的模型参数更新全局模型参数
            if parameters_prime:
                self.parameters = parameters_prime
```

可以看到单轮的联邦学习模型训练就是通过 `fit_round()` 实现，具体的代码如下所示：

```python
def fit_round(
    self,
    server_round: int,
    timeout: Optional[float],
) -> Optional[
    Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
]:
    # 使用优化策略（默认是FedAvg），获取客户端及其相应的指令
    client_instructions = self.strategy.configure_fit(
        server_round=server_round,
        parameters=self.parameters,
        client_manager=self._client_manager,
    )

    # 客户端基于本地数据进行模型训练
    results, failures = fit_clients(
        client_instructions=client_instructions,
        max_workers=self.max_workers,
        timeout=timeout,
    )

    # 聚合客户端发来的模型参数
    aggregated_result: Tuple[
        Optional[Parameters],
        Dict[str, Scalar],
    ] = self.strategy.aggregate_fit(server_round, results, failures)

    parameters_aggregated, metrics_aggregated = aggregated_result
    return parameters_aggregated, metrics_aggregated, (results, failures)
```

下面具体 `fit_clients()` 是如何发起模型训练的：

```python
def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    # 多线程并发调用 fit_client 方法实现客户端模型训练
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,
        )

    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures

# 通过 ClientProxy 发起客户端的 fit() 模型训练，ins 中包含全局模型参数
def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res
```

### 4.发送全局更新结果

还没找到相关代码，不知道是如何做的，只找到了如何在client中receive（message_handler的 maybe_call_fit函数）



## 小计