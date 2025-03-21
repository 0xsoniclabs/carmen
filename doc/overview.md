# How to Build
1. Clone the repository
   ```
   git clone https://github.com/0xsoniclabs/carmen
   ```
2. Run tests
    * [Go tests](https://github.com/0xsoniclabs/carmen/tree/main/go#development)
    * [C++ tests](https://github.com/0xsoniclabs/carmen/blob/main/cpp/README.md#build-and-test)
***

# How to Integrate
1. Get the latest `Go` version of `Carmen`
```
go get -u github.com/0xsoniclabs/carmen/go
```
2. Import the public interface
```
import "github.com/0xsoniclabs/carmen/go/carmen"
```


# How to Use
Schema S5 is the only schema compatible with Merkle-Patricia Tries. Carmen enables always LiveDB to keep the state of the last block.
ArchiveDB can be enabled to retain historical data. If ArchiveDB is enabled, the state of the last block is appended to the archive for later retrieval.

The Carmen interface is loosely related to the go-ethereum's StateDB interface, though it differs w.r.t. block management and transaction management.
The functionality contains creating, opening, and closing an instance of Carmen.
In addition, we have getter/setter operations for the state (Balance/Nonce/Code/Storage)
and other operations related to scoping (e.g., Snapshots).

## LiveDB

Below is an example that creates a new account and adds 100 units to the balance:

<details>
    <summary>Functional style API</summary>

```
func ExampleDatabase_AddBlock() {
	dir, err := os.MkdirTemp("", "carmen_db_*")
	if err != nil {
		log.Fatalf("cannot create temporary directory: %v", err)
	}
	db, err := carmen.OpenDatabase(dir, carmen.GetCarmenGoS5WithoutArchiveConfiguration(), nil)
	if err != nil {
		log.Fatal(err)
	}
	// Add a new block
	if err := db.AddBlock(5, func(context carmen.HeadBlockContext) error {
		if err := context.RunTransaction(func(context carmen.TransactionContext) error {
			context.CreateAccount(carmen.Address{1})
			context.AddBalance(carmen.Address{1}, big.NewInt(100))
			fmt.Printf("Transaction executed")
			return nil
		}); err != nil {
			log.Fatalf("cannot create transaction: %v", err)
		}
		return nil
	}); err != nil {
		log.Fatalf("cannot add block: %v", err)
	}
	if err := db.Close(); err != nil {
		log.Fatalf("cannot close db: %v", err)
	}
	// Output: Transaction executed
}
```
</details>

<details>
    <summary>Imperative style API</summary>

```
func ExampleDatabase_BeginBlock() {
	dir, err := os.MkdirTemp("", "carmen_db_*")
	if err != nil {
		log.Fatalf("cannot create temporary directory: %v", err)
	}
	db, err := carmen.OpenDatabase(dir, carmen.GetCarmenGoS5WithoutArchiveConfiguration(), nil)
	if err != nil {
		log.Fatal(err)
	}
	// Begin a new block
	bctx, err := db.BeginBlock(5)
	if err != nil {
		log.Fatalf("cannot begin block: %v", err)
	}
	// Begin a new transaction within the block
	tctx, err := bctx.BeginTransaction()
	if err != nil {
		log.Fatalf("cannot begin transaction: %v", err)
	}
	tctx.CreateAccount(carmen.Address{1})
	tctx.AddBalance(carmen.Address{1}, big.NewInt(100))
	if err := tctx.Commit(); err != nil {
		log.Fatalf("cannot commit transaction: %v", err)
	}
	if err := bctx.Commit(); err != nil {
		log.Fatalf("cannot commit block: %v", err)
	}
	if err := db.Close(); err != nil {
		log.Fatalf("cannot close db: %v", err)
	}
	if err := os.RemoveAll(dir); err != nil {
		log.Fatalf("cannot remove dir: %v", err)
	}
}
```
</details>


## Archive

A historical state can be queried on previously stored blocks. Two forms of APIs are provided: \

<details>
    <summary>Functional style API</summary>

```
func ExampleDatabase_QueryBlock() {
	dir, err := os.MkdirTemp("", "carmen_db_*")
	if err != nil {
		log.Fatalf("cannot create temporary directory: %v", err)
	}
	db, err := carmen.OpenDatabase(dir, carmen.GetCarmenGoS5WithArchiveConfiguration(), nil)
	if err != nil {
		log.Fatal(err)
	}
	// Add a new block
	if err := db.AddBlock(5, func(context carmen.HeadBlockContext) error {
		if err := context.RunTransaction(func(context carmen.TransactionContext) error {
			context.CreateAccount(carmen.Address{1})
			context.AddBalance(carmen.Address{1}, big.NewInt(100))
			return nil
		}); err != nil {
			log.Fatalf("cannot create transaction: %v", err)
		}
		return nil
	}); err != nil {
		log.Fatalf("cannot add block: %v", err)
	}
	// block wait until the archive is in sync
	if err := db.Flush(); err != nil {
		log.Fatalf("cannot flush: %v", err)
	}
	// query history block
	if err := db.QueryBlock(5, func(ctxt carmen.HistoricBlockContext) error {
		return ctxt.RunTransaction(func(ctxt carmen.TransactionContext) error {
			balance := ctxt.GetBalance(carmen.Address{1})
			if got, want := balance, big.NewInt(100); got.Cmp(want) != 0 {
				log.Fatalf("balance does not match: %d != %d", got, want)
			}
			fmt.Printf("Balance of %v is %d\n", carmen.Address{1}, balance)
			return nil
		})
	}); err != nil {
		log.Fatalf("cannot query block: %v", err)
	}
	if err := db.Close(); err != nil {
		log.Fatalf("cannot close db: %v", err)
	}
	if err := os.RemoveAll(dir); err != nil {
		log.Fatalf("cannot remove dir: %v", err)
	}
	// Output: Balance of [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] is 100
}
```
</details>

<details>
    <summary>Imperative style API</summary>

```
func ExampleDatabase_GetHistoricContext() {
	dir, err := os.MkdirTemp("", "carmen_db_*")
	if err != nil {
		log.Fatalf("cannot create temporary directory: %v", err)
	}
	db, err := carmen.OpenDatabase(dir, carmen.GetCarmenGoS5WithArchiveConfiguration(), nil)
	if err != nil {
		log.Fatal(err)
	}
	// Add a new block
	if err := db.AddBlock(5, func(context carmen.HeadBlockContext) error {
		if err := context.RunTransaction(func(context carmen.TransactionContext) error {
			context.CreateAccount(carmen.Address{1})
			context.AddBalance(carmen.Address{1}, big.NewInt(100))
			return nil
		}); err != nil {
			log.Fatalf("cannot create transaction: %v", err)
		}
		return nil
	}); err != nil {
		log.Fatalf("cannot add block: %v", err)
	}
	// block wait until the archive is in sync
	if err := db.Flush(); err != nil {
		log.Fatalf("cannot flush: %v", err)
	}
	// query history block
	hctx, err := db.GetHistoricContext(5)
	if err != nil {
		log.Fatalf("cannot begin history query: %v", hctx)
	}
	tctx, err := hctx.BeginTransaction()
	if err != nil {
		log.Fatalf("cannot begin transaction: %v", err)
	}
	balance := tctx.GetBalance(carmen.Address{1})
	if got, want := balance, big.NewInt(100); got.Cmp(want) != 0 {
		log.Fatalf("balance does not match: %d != %d", got, want)
	}
	fmt.Printf("Balance of %v is %d\n", carmen.Address{1}, balance)
	if err := tctx.Abort(); err != nil {
		log.Fatalf("cannot abort transaction: %v", err)
	}
	if err := hctx.Close(); err != nil {
		log.Fatalf("cannot close block: %v", err)
	}
	if err := db.Close(); err != nil {
		log.Fatalf("cannot close db: %v", err)
	}
	if err := os.RemoveAll(dir); err != nil {
		log.Fatalf("cannot remove dir: %v", err)
	}
	// Output: Balance of [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] is 100
}
```
</details>


## Query API
LiveDB and Archive can be queried using a **query API** providing a functional style callback as shown below:

<details>
    <summary>Querying a LiveDB</summary>

```
func ExampleDatabase_QueryHeadState() {
	dir, err := os.MkdirTemp("", "carmen_db_*")
	if err != nil {
		log.Fatalf("cannot create temporary directory: %v", err)
	}
	db, err := carmen.OpenDatabase(dir, carmen.GetCarmenGoS5WithArchiveConfiguration(), nil)
	if err != nil {
		log.Fatal(err)
	}
	// Query state information for the current head block
	if err := db.QueryHeadState(func(context carmen.QueryContext) {
		balance := context.GetBalance(carmen.Address{1, 2, 3})
		fmt.Printf("Account balance: %v", balance)
	}); err != nil {
		log.Fatalf("query operation failed: %v", err)
	}
	if err := db.Close(); err != nil {
		log.Fatalf("cannot close db: %v", err)
	}
	// Output: Account balance: 0
}
```
</details>

<details>
    <summary>Querying an Archive</summary>

```
func ExampleDatabase_QueryHistoricState() {
	dir, err := os.MkdirTemp("", "carmen_db_*")
	if err != nil {
		log.Fatalf("cannot create temporary directory: %v", err)
	}
	db, err := carmen.OpenDatabase(dir, carmen.GetCarmenGoS5WithArchiveConfiguration(), nil)
	if err != nil {
		log.Fatal(err)
	}
	// Add a new block
	if err := db.AddBlock(5, func(context carmen.HeadBlockContext) error {
		if err := context.RunTransaction(func(context carmen.TransactionContext) error {
			context.CreateAccount(carmen.Address{1, 2, 3})
			context.AddBalance(carmen.Address{1, 2, 3}, big.NewInt(100))
			return nil
		}); err != nil {
			log.Fatalf("cannot create transaction: %v", err)
		}
		return nil
	}); err != nil {
		log.Fatalf("cannot add block: %v", err)
	}
	// block wait until the archive is in sync
	if err := db.Flush(); err != nil {
		log.Fatalf("cannot flush: %v", err)
	}
	// Query state information for the current head block
	if err := db.QueryHistoricState(5, func(context carmen.QueryContext) {
		balance := context.GetBalance(carmen.Address{1, 2, 3})
		if got, want := balance, big.NewInt(100); got.Cmp(want) != 0 {
			log.Fatalf("balance does not match: %d != %d", got, want)
		}
		fmt.Printf("Balance of %v is %d\n", carmen.Address{1, 2, 3}, balance)
	}); err != nil {
		log.Fatalf("query operation failed: %v", err)
	}
	if err := db.Close(); err != nil {
		log.Fatalf("cannot close db: %v", err)
	}
	// Output: Balance of [1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] is 100
}
```
</details>
