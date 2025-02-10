# 部署与调用合约

在上节我们已编译好一个NEP17合约文件（NEP17.nef ）和合约描述文件（NEP17.manifest.json ），本节我们将使用Neo-CLI部署并调用该合约。

## 部署合约

在 Neo-CLI 中，打开前面在 [准备工作](prerequisites.md) 里创建的钱包 0.json，输入部署合约命令  `deploy <nefFilePath> [manifestFile]` ，例如：

```bash
deploy NEP17.nef
```

或

```bash
deploy NEP17.nef NEP17.manifest.json
```

输入命令后，程序会部署NEP-17合约并且自动支付手续费。

```bash
neo> deploy NEP17.nef
Script hash: 0xb7f4d011241ec13db16c0e3484bdd5dd9a536f26
Gas: 10.1477624

Signed and relayed transaction with hash=0xe03aade81fb96c44e115a1cc9cfe984a9df4a283bd10aa0aefa7ebf3e296f757
```

更多部署信息请参考 [部署智能合约](../develop/deploy/deploy.md)。

## 调用合约

部署完成后，使用 invoke 命令调用合约：

```bash
invoke <scriptHash> <operation> [contractParameters=null] [witnessAddress=null]
```

例如：

```bash
invoke 0xb7f4d011241ec13db16c0e3484bdd5dd9a536f26 symbol
```

成功执行后，屏幕输出如下信息：

```bash
Invoking script with: '10c00c046e616d650c14f9f81497c3f9b62ba93f73c711d41b1eeff50c2341627d5b52'
VM State: HALT
Gas Consumed: 0.0103609
Evaluation Stack: [{"type":"ByteArray","value":"VG9rZW5TeW1ib2w="}]

relay tx(no|yes):
```

其中：

- VM State:  `HALT` 表示虚拟机执行成功， `FAULT` 表示虚拟机执行时遇到异常退出。
- Evaluation Stack: 合约执行结果，如果 value 是字符串或 ByteArray，则是 Base64 编码后的结果。
- 你可以在 [这里](https://neo.org/converter/) 进行数据格式转换 `VG9rZW5TeW1ib2w=` => `TokenSymbol`。

更多智能合调用信息请参考 [调用智能合约](../develop/deploy/invoke.md)。
