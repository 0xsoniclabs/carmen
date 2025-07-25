// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

package ticker

import "time"

//go:generate mockgen -source ticker.go -destination ticker_mocks.go -package ticker

// Ticker is an abstraction of a ticker from standard time package.
// It contains a channel which produces ticks at certain intervals
// defined by implementations. For example, TimeTicker wraps
// the standard time.Ticker based on time.Duration intervals.
// Furthermore, the ticker includes the method Stop to terminate
// the ticker. When the ticker is stopped, no more ticks will be
// sent via the channel. The Stop method should be called all
// the time after the ticker is no longer necessary, even if further
// ticking does not matter, because Stop also makes sure all the
// resources are freed up.
type Ticker interface {

	// C returns the channel on which the ticks are delivered.
	C() <-chan time.Time

	// Stop turns off a ticker. After Stop, no more ticks will be sent.
	Stop()
}

// TimeTicker is a wrapper around time.Ticker, which is a Ticker
// implementation based on the standard library.
type TimeTicker struct {
	ticker *time.Ticker
}

// NewTimeTicker creates a new TimeTicker, which is a Ticker
// implementation based on the standard time.Ticker.
func NewTimeTicker(d time.Duration) TimeTicker {
	return TimeTicker{time.NewTicker(d)}
}

func (t TimeTicker) C() <-chan time.Time {
	return t.ticker.C
}

func (t TimeTicker) Stop() {
	t.ticker.Stop()
}
