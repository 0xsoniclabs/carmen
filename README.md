Sonic-DB(Carmen) is licensed under the [Business Source License (BSL)](LICENSE). 
The BSL prohibits Sonic-DB (Carmen) from being used in production by any other project. 
Anyone can view or use the licensed code for internal or testing purposes. 
Still, commercial use is limited to Sonic Operations Ltd and Sonic users operating on Sonic's mainnet and/or testnet.

Carmen is a fast and space-conservative database for blockchains. 
It outperforms other projects in transaction speed and space consumption. 
Carmen assumes a linear evolution of blocks and supports two variants of databases: 
LiveDB, which keeps the last state of the last block only, and ArchiveDB, 
which keeps all states over all blocks. 
The storage layer is abstracted, and various schemas have been implemented 
to read and write information in memory, a key-value store (LevelDB, etc.), 
or a native file format. 
One of the schemas is a Merkle-Patricia Trie (MPT) compatible with the EVM. 
All schemas are implemented in Go, while some are in C++.

More information: [How to use](doc/overview.md)