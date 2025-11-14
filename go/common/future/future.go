package future

type Promise[T any] chan<- Result[T]
type Future[T any] <-chan Result[T]

func ImmediateOk[T any](value T) <-chan Result[T] {
	ch := make(chan Result[T], 1)
	ch <- Ok(value)
	close(ch)
	return ch
}

func ImmediateErr[T any](err error) <-chan Result[T] {
	ch := make(chan Result[T], 1)
	ch <- Err[T](err)
	close(ch)
	return ch
}

func (f Future[T]) Await() (T, error) {
	res, ok := <-f
	if !ok {
		var zero T
		return zero, nil
	}
	return res.Get()
}
