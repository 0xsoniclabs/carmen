package future

// Result encapsulates a value along with an error. It is intended to be used
// in scenarios where a single type is needed to represent the outcome of an
// operation that can either succeed with a value of type T or fail with an
// error. This may, for instance, be useful for channels or containers.
type Result[T any] struct {
	Value T
	Error error
}

func Ok[T any](value T) Result[T] {
	return Result[T]{Value: value}
}

func Err[T any](err error) Result[T] {
	return Result[T]{Error: err}
}

// Get returns the value and error contained in the Result.
func (r Result[T]) Get() (T, error) {
	return r.Value, r.Error
}
