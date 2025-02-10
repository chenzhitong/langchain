# 搭建本地网络

## 搭建私链

Neo 官方提供了供用户开发、调试和测试的测试网（Test Net），但在本地搭建你自己的私链将获得更多的灵活性以及取之不尽的测试币。你可以选择以下一种方式搭建私有链并提取创世区块中的 NEO 和 GAS：

- [使用单节点搭建](../develop/network/private-chain/solo.md)
- [使用多节点搭建](../develop/network/private-chain/private-chain2.md)

## 准备节点钱包文件

现在我们创建一个新的钱包文件用于发布智能合约：

1. 在 Neo-CLI 中，输入命令 `create wallet`， 创建一个新的钱包文件 0.json，复制默认地址备用。
2. 打开前面提取了 NEO 和 GAS 的钱包，将钱包中全部的资产都转入钱包 0.json 中，等待交易确认。
3. 打开钱包文件 0.json，可以看到其中的资产。

更多 Neo-CLI 命令，请参考 [CLI](../node/cli/cli.md)。
