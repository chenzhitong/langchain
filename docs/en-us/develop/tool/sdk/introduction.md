# About RpcClient

`RpcClient` is a C# dependency library that can be used to develop various applications on Neo, such as wallet clients, games, etc. This project can help you easily invoke Neo RPC interfaces, construct transactions, and invoke contracts.

This document is applicable to Neo N3. It is recommended you use NEO SDK in conjunction with Visual Studio 2019.  

## Main features

- Built-in RPC invocation methods.

- Transaction construction, serialization and deserialization

- Script construction for smart contracts

- Easy-to-use wallet functions: transfer, balance inquiry, and GAS Claim method

## Adding reference

1. In Visual Studio 2019, create a new .NET project

   > [!Note]
   >
   > The .NET version of the project cannot be earlier than the .NET version used by the Neo SDK.

2. Right-click the project and select `Manage NuGet Packages`.

3. Search for `Neo.Network.RPC.RpcClient` in the search box and install the plugin that appears.

4. Add the following code to the desired project file header:

   ```c#
   using Neo.Network.RPC;
   ```

> [!Note]
>
> If you use SDK to construct a transaction that requires a signature, you need to ensure that the RpcClient obeject and the network it is connected to are configured the same way, or the transaction constructed by the SDK will not be validated in the blockchain. To do so, load Neo-CLI config.json when constructing the RpcClient object, for example:
>
> RpcClient client = new RpcClient(new Uri("http://localhost:20332"), null, null, ProtocolSettings.Load("config.json"))

## Exception handing

`RpcClient` mainly interacts with Neo nodes through RPC requests. When the message returned by the RPC request contains an Error the system throws an exception. The most common exception type is  `RpcException`, which includes:

- -100, "Unknown transaction/blockhash"
- -300, "Insufficient funds"
- -301, "The necessary fee is more than the Max_fee, this transaction is failed. Please increase your Max_fee value."
- -400, "Access denied"
- -500, Relay does not succeed, the detailed reasons contain "AlreadyExists, OutOfMemory, UnableToVerify, Invalid, Expired, InsufficientFunds, PolicyFail, Unknown"
- -32600, "Invalid Request"
- -32601, "Method not found"
- -32602, "Invalid params"
- -32700, "Parse error"

Additionally, you may encounter other exceptions built in the .NET platform, such as  `ArgumentNullException` and `FormatException` when the parameters passed in are null or incorrectly formatted. Therefore, reasonable exception capture and prompts are required when using the SDK.

## Project repository

The `RpcClient` is a functional subset of the neo-modules project. For complete open source projects refer to:

Neo-modules：https://github.com/neo-project/neo-modules

Welcome to raise any problems you encountered in practice in the project issue list:

https://github.com/neo-project/neo-modules/issues
