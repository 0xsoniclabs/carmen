//go:build !carmen_rust && !carmen_cpp

package externalstate

func init() {
	// No external state configurations are supported when the "externalstate" build tag is not set.
}
