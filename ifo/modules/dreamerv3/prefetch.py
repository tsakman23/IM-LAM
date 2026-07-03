import queue
import threading
from typing import Any, Callable, Iterator, Optional, Union


class Prefetch:
    """Asynchronously prefetch items from a source stream.

    Spawns a worker thread that fetches items ahead of consumption and buffers
    them in a queue to overlap I/O or compute with downstream processing.
    """

    def __init__(
        self,
        source: Union[Iterator[Any], Callable[[Any], Any]],
        prefetch_factor: int = 1,
    ) -> None:
        """Create a prefetching stream.

        Args:
            source (Union[Iterator[Any], Callable[[Any], Any]]): Iterable or callable to prefetch.
            prefetch_factor (int): Maximum number of in-flight prefetched items.

        Raises:
            ValueError: If prefetch_factor is less than 1.
        """
        if prefetch_factor < 1:
            raise ValueError("Amount must be at least 1")

        # Wrap callable sources that return single values
        if callable(source) and not hasattr(source, "__iter__"):
            # Wrap the callable to yield values infinitely
            def _wrapper():
                while True:
                    yield source()

            self._source_input = _wrapper
        else:
            self._source_input = source

        self.source: Optional[Iterator[Any]] = None

        # Initialize state and synchronization primitives
        self.state = None
        self.requests = threading.Semaphore(prefetch_factor)
        self.prefetch_factor = prefetch_factor
        self.queue = queue.Queue()

        # Create and manage worker thread
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.started = False

    def __iter__(self) -> "Prefetch":
        """Return the iterator.

        Returns:
            Self, allowing the object to be used as an iterator.
        """
        return self

    def __next__(self) -> Any:
        """Return the next item, blocking if the queue is empty.

        The source iterator and worker thread are started lazily on the first call to this method.

        Returns:
            The next item from the source.

        Raises:
            RuntimeError: If an error occurred in the worker thread.
            StopIteration: If the source iterator is exhausted.
        """
        if not self.started:
            # Initialize the source iterator
            self.source = iter(self._source_input) if hasattr(self._source_input, "__iter__") else self._source_input()
            self.state = self._getstate()

            # Start the worker thread
            self.worker.start()
            self.started = True

        result = self.queue.get()
        self.requests.release()

        # Check if the result is an error message (string)
        if isinstance(result, str):
            raise RuntimeError(result)

        # Unpack the data and state
        data, self.state = result
        return data

    def save(self) -> Optional[Any]:
        """Return the last saved state of the source stream, if any.

        Returns:
            The current state of the source stream, or None if the source
            doesn't support state saving.
        """
        return self.state

    def load(self, state: Optional[Any]) -> None:
        """Load state into the source stream and reset prefetch bookkeeping.

        Args:
            state: State to load into the source stream.

        Raises:
            RuntimeError: If the source doesn't support state loading or hasn't been initialized.
        """
        if not self.started:
            raise RuntimeError("Cannot load state before the source iterator has been initialized")

        if self.started:
            # Clear the queue of any pending items
            for _ in range(self.prefetch_factor):
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

        # Load state into the source if it supports it
        if hasattr(self.source, "load"):
            self.source.load(state)
        elif state is not None:
            raise RuntimeError("Source doesn't support state loading")

        if self.started:
            # Release semaphore to allow prefetching to resume
            self.requests.release(self.prefetch_factor)

    def _worker(self) -> None:
        """Worker thread body that fetches and queues items.

        This method runs in a separate thread and continuously:
        1. Acquires a semaphore permit
        2. Fetches the next item from the source
        3. Captures the current state
        4. Queues the result

        If an exception occurs, it queues an error message as a string.
        """
        try:
            while True:
                self.requests.acquire()
                data = next(self.source)
                state = self._getstate()
                self.queue.put((data, state))
        except StopIteration:
            # Source is exhausted - this is normal
            pass
        except Exception as e:
            # Queue the error message for the main thread to handle
            self.queue.put(str(e))
            raise

    def _getstate(self) -> Optional[Any]:
        """Best-effort snapshot of the source state for checkpointing.

        Returns:
            The current state of the source stream if it supports saving,
            otherwise None.
        """
        if self.source is not None and hasattr(self.source, "save"):
            return self.source.save()
        else:
            return None
