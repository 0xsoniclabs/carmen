The content of this directory was copied from the Ethereum release v1.16.9, as
this was the last release including a Verkle Trie implementation.

On the go-ethereum main branch, Verkle Tries got removed by
https://github.com/ethereum/go-ethereum/commit/3f641dba872dd43c8232b9384b4c09f0b9e3bd96
and starting with v1.17 are no longer part of the release.

To preserve this implementation for comparison in Carmen, and to avoid version
conflicts towards go-ethereum dependencies in down-stream projects, the code
has been copied here.
