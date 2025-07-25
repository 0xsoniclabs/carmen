// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package interrupt

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/0xsoniclabs/carmen/go/common"
)

const ErrCanceled = common.ConstError("interrupted")

// IsCancelled returns true if the given context's CancelFunc has been called.
// Otherwise, returns false.
func IsCancelled(ctx context.Context) bool {
	select {
	case <-ctx.Done():
		return true
	default:
		return false
	}
}

// CancelOnInterrupt catches SIGTERM and SIGINT and cancels the returned context.
// After first signal is caught and cancel is called, this thread is closed and
// no more signals are getting called, hence any other interrupt kills the app.
func CancelOnInterrupt(parent context.Context) context.Context {
	ctx, cancel := context.WithCancel(parent)
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		defer signal.Stop(c)
		select {
		case <-c:
			log.Println("shutting down")
			cancel()
		case <-ctx.Done():
		}
	}()

	return ctx
}
