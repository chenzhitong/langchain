# getwalletunclaimedgas 方法

显示钱包中未提取的 GasToken 数量。

> [!Note]
>
> - 执行此命令前需要 RPC 调用 openwallet 方法来打开钱包。
> - 此方法由插件提供，需要安装 [RpcServer](https://github.com/neo-project/neo-modules/releases) 插件才可以调用

## 调用示例

请求正文：

```json
{
  "jsonrpc": "2.0",
  "method": "getwalletunclaimedgas",
  "params": ["NgaiKFjurmNmiRzDRQGs44yzByXuSkdGPF"],
  "id": 1
}
```

响应正文：

```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": "750000000"
}
```

响应说明：

返回未提取的 GasToken 数量。
