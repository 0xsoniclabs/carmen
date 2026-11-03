package state

// Revision identifies the fork rules applied when finalizing a transaction.
type Revision uint16

const (
	CANCUN Revision = iota
	AMSTERDAM
	numRevisions
)

func LatestRevision() Revision { return numRevisions - 1 }
