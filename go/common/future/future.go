package future

type Promise[T any] chan<- Result[T]
type Future[T any] <-chan Result[T]

func Create[T any]() (Promise[T], Future[T]) {
	ch := make(chan Result[T], 1)
	return Promise[T](ch), Future[T](ch)
}

func ImmediateOk[T any](value T) Future[T] {
	ch := make(chan Result[T], 1)
	ch <- Ok(value)
	close(ch)
	return ch
}

func ImmediateErr[T any](err error) Future[T] {
	ch := make(chan Result[T], 1)
	ch <- Err[T](err)
	close(ch)
	return ch
}

func (p Promise[T]) Forward(f Future[T]) {
	go func() {
		p <- <-f
		close(p)
	}()
}

func (p Promise[T]) Fulfill(result Result[T]) {
	p <- result
	close(p)
}

func (f Future[T]) Get() Result[T] {
	return <-f
}

func (f Future[T]) Await() (T, error) {
	return f.Get().Get()
}

func Then[A, B any](f Future[A], transform func(A) (B, error)) Future[B] {
	promise, future := Create[B]()
	go func() {
		result := f.Get()
		if result.Error != nil {
			promise.Fulfill(Err[B](result.Error))
			return
		}
		value, err := transform(result.Value)
		if err != nil {
			promise.Fulfill(Err[B](err))
			return
		}
		promise.Fulfill(Ok(value))
	}()
	return future
}
