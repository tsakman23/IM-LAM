import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ifo.modules.dreamerv3.utils import UUID, Chunk, RWLock, Uniform


class ReplayBuffer:
    """
    Efficient replay buffer for storing and sampling experience sequences.

    The buffer stores experience in chunks for efficient memory management and
    supports uniform sampling. Data can be persisted to disk and
    loaded back for training continuation.
    """

    def __init__(
        self,
        length: int,
        capacity: Optional[int] = None,
        directory: Optional[str] = None,
        chunksize: int = 1024,
        online: bool = False,
        selector: Optional[Any] = None,
        save_wait: bool = False,
        seed: int = 0,
    ):
        """
        Initialize the replay buffer.

        Args:
            length: Length of experience sequences to sample (number of time steps).
            capacity: Maximum number of sequences to store. If None, buffer grows unbounded.
            directory: Directory path for saving/loading buffer data. If None, no persistence.
            chunksize: Number of steps per chunk for internal storage optimization.
            online: If True, enables online sampling mode for prioritizing recent sequences.
            selector: Custom sampling strategy (e.g., prioritized replay). Defaults to uniform.
            save_wait: If True, wait for all save operations to complete before returning.
            seed: Random seed for the default uniform selector.
        """
        # Configuration parameters
        self.length = length  # Sequence length for sampling
        self.capacity = capacity  # Maximum buffer capacity
        self.chunksize = chunksize  # Size of each storage chunk

        # Sampling strategy
        self.sampler: Any = selector or Uniform(seed)

        # Core storage data structures
        self.chunks = {}  # UUID -> Chunk mapping for all stored chunks
        # Reference counts for each chunk (for garbage collection)
        self.refs = {}
        self.refs_lock = threading.RLock()  # Lock for thread-safe reference counting

        # Sequence tracking
        # itemid -> (chunkid, index) mapping for all stored sequences
        self.items = {}
        self.fifo = deque()  # FIFO queue of item IDs for capacity management
        self.itemid = 0  # Counter for assigning unique item IDs

        # Active data collection state
        # worker -> (chunkid, index) mapping for current write position
        self.current = {}
        # worker -> deque of (chunkid, index) pairs
        self.streams = defaultdict(deque)
        self.rwlock = RWLock()  # Read-write lock for thread-safe access

        # Online mode configuration
        self.online = online
        if online:
            self.lengths = defaultdict(int)  # Track sequence lengths per worker
            self.queue = deque()  # Queue for online sequences

        # Persistence configuration
        if directory:
            self.directory = Path(directory)
            self.directory.mkdir(parents=True, exist_ok=True)
            self.workers = ThreadPoolExecutor(16, "replay_saver")  # Thread pool for async saves
            self.saved = set()  # Track which chunks have been saved to disk
        else:
            self.directory = None
        self.save_wait = save_wait  # Whether to block on save operations

        # Performance metrics
        self.metrics = {"samples": 0, "inserts": 0, "updates": 0}

    def __len__(self) -> int:
        """Return the number of sequences currently stored in the buffer."""
        return len(self.items)

    def stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics and reset metric counters.

        Returns:
            Dictionary containing buffer statistics including:
            - samples: Number of sample operations performed
            - inserts: Number of sequence insertions
            - updates: Number of sequence updates
            - size: Current number of stored sequences
            - chunks: Number of chunks in memory
        """
        stats = dict(self.metrics)
        stats.update({"size": len(self.items), "chunks": len(self.chunks)})
        self.metrics = {"samples": 0, "inserts": 0, "updates": 0}
        return stats

    def add(self, step: Dict[str, np.ndarray], worker: int = 0) -> None:
        """
        Add a single step of experience to the buffer.

        Steps are accumulated into sequences of length `self.length` before
        being made available for sampling. Multiple workers can add data
        concurrently using different worker IDs.

        Args:
            step: Dictionary of arrays representing one environment step.
                 Keys starting with 'log/' are filtered out before storage.
            worker: Worker ID for concurrent data collection. Defaults to 0.
        """
        # Filter out logging keys to save memory
        step = {k: v for k, v in step.items() if not k.startswith("log/")}

        with self.rwlock.reading:
            # Convert all values to numpy arrays for consistent storage
            step = {k: np.asarray(v) for k, v in step.items()}

            # Get or create current chunk for this worker
            if worker not in self.current:
                chunk = Chunk(self.chunksize)
                with self.refs_lock:
                    self.refs[chunk.uuid] = 1
                self.chunks[chunk.uuid] = chunk
                self.current[worker] = (chunk.uuid, 0)

            chunkid, index = self.current[worker]
            # Add step ID for tracking (matches official implementation)
            step["stepid"] = np.frombuffer(bytes(chunkid) + index.to_bytes(4, "big"), np.uint8)
            stream = self.streams[worker]
            chunk = self.chunks[chunkid]
            assert chunk.length == index, (chunk.length, index)
            chunk.append(step)
            assert chunk.length == index + 1, (chunk.length, index + 1)
            stream.append((chunkid, index))
            with self.refs_lock:
                self.refs[chunkid] += 1

            index += 1
            if index < chunk.size:
                self.current[worker] = (chunkid, index)
            else:
                self._complete(chunk, worker)

            assert len(self.streams) == len(self.current)

            # Once we have enough steps, insert the sequence into the buffer
            if len(stream) >= self.length:
                # Increment is not thread safe thus inaccurate but faster than
                # locking.
                self.metrics["inserts"] += 1
                chunkid, index = stream.popleft()
                self._insert(chunkid, index)

                # Add to online queue if online mode is enabled and at sequence boundary
                if self.online and self.lengths[worker] % self.length == 0:
                    self.queue.append((chunkid, index))

            # Update sequence length counter for online mode
            if self.online:
                self.lengths[worker] += 1

    def sample(self, batch: int, mode: str = "train") -> Dict[str, np.ndarray]:
        """
        Sample a batch of experience sequences from the buffer.

        Args:
            batch: Number of sequences to sample.
            mode: Sampling mode - 'train', 'report', or 'eval'. Only 'train' updates metrics.

        Returns:
            Dictionary mapping keys to batched arrays of shape (batch, length, ...).
            Includes an 'is_first' flag marking sequence starts.
        """
        if len(self.sampler) == 0:
            raise ValueError("Replay buffer is empty")

        # Sample individual sequences
        seqs, is_online = zip(*[self._sample(mode) for _ in range(batch)])

        # Assemble sequences into a batch
        data = self._assemble_batch(list(seqs), 0, self.length)

        # Annotate with metadata (is_first flags, etc.)
        data = self._annotate_batch(data, is_online, True)
        return data

    def update(self, data: Dict[str, np.ndarray]) -> None:
        """
        Update stored sequences with new values (e.g., for off-policy corrections).

        Args:
            data: Dictionary containing:
                - 'stepid': Array of step identifiers for sequences to update
                - 'priority': Optional priority values for prioritized replay
                - Other keys: New values to update in the sequences
        """
        stepid = data.pop("stepid")
        priority = data.pop("priority", None)

        # Update priorities if using prioritized replay
        if priority is not None:
            assert priority.ndim == 2, priority.shape
            # Check if sampler supports prioritization
            if hasattr(self.sampler, "prioritize"):
                self.sampler.prioritize(stepid.reshape((-1, stepid.shape[-1])), priority.flatten())

        # Update sequence values in the buffer
        if data:
            for i, stepid in enumerate(stepid):
                # Decode step ID to get chunk and index
                stepid = stepid[0].tobytes()
                chunkid = UUID(stepid[:-4])
                index = int.from_bytes(stepid[-4:], "big")
                values = {k: v[i] for k, v in data.items()}
                try:
                    self._setseq(chunkid, index, values)
                except KeyError:
                    # Chunk may have been evicted due to capacity limits
                    pass

    def _sample(self, mode: str) -> tuple:
        """
        Sample a single sequence from the buffer.

        Args:
            mode: Sampling mode ('train', 'report', or 'eval').

        Returns:
            Tuple of (sequence_dict, is_online_flag).
        """
        assert mode in ("train", "report", "eval"), mode
        if mode == "train":
            # Increment is not thread safe thus inaccurate but faster than
            # locking.
            self.metrics["samples"] += 1

        while True:
            try:
                # Prioritize online queue for training mode when available
                if self.online and self.queue and mode == "train":
                    chunkid, index = self.queue.popleft()
                    is_online = True
                else:
                    # Use sampler to select a sequence (uniform or prioritized)
                    itemid = self.sampler()
                    chunkid, index = self.items[itemid]
                    is_online = False

                # Retrieve the sequence data
                seq = self._getseq(chunkid, index, concat=False)
                return seq, is_online
            except KeyError:
                # Sequence may have been removed; retry
                continue

    def _insert(self, chunkid: UUID, index: int) -> None:
        """
        Insert a sequence into the buffer, making it available for sampling.

        Args:
            chunkid: UUID of the chunk containing the sequence start.
            index: Index within the chunk where the sequence starts.
        """
        # Evict old sequences if at capacity
        while self.capacity and len(self.items) >= self.capacity:
            self._remove()

        # Create new item entry
        itemid = self.itemid
        self.itemid += 1
        self.items[itemid] = (chunkid, index)

        # Add to FIFO queue for capacity management
        self.fifo.append(itemid)

        # Get stepids for the sequence and add to sampler
        stepids = self._getseq(chunkid, index, ["stepid"])["stepid"]
        self.sampler[itemid] = stepids

    def _remove(self) -> None:
        """Remove the oldest sequence from the buffer to make space."""
        # Remove from FIFO queue
        itemid = self.fifo.popleft()
        chunkid, index = self.items.pop(itemid)

        # Remove from sampler
        del self.sampler[itemid]

        # Decrement reference count and potentially garbage collect chunk
        with self.refs_lock:
            self.refs[chunkid] -= 1
            if self.refs[chunkid] <= 0:
                # No more references to this chunk, remove it
                del self.refs[chunkid]
                chunk = self.chunks.pop(chunkid)
                # Also decrement reference to successor chunk
                if chunk.succ in self.refs:
                    self.refs[chunk.succ] -= 1

    def _getseq(
        self, chunkid: UUID, index: int, keys: Optional[List[str]] = None, concat: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve a sequence of length `self.length` from the buffer.

        Sequences may span multiple chunks, in which case they are assembled
        by following successor chunk pointers.

        Args:
            chunkid: UUID of the chunk containing the sequence start.
            index: Starting index within the chunk.
            keys: Optional list of keys to retrieve. If None, gets all keys.
            concat: If True, concatenate arrays. If False, return list of parts.

        Returns:
            Dictionary mapping keys to arrays (if concat=True) or lists of arrays.
        """
        parts = []
        remaining = self.length
        current_chunk = chunkid
        current_index = index

        while remaining > 0:
            chunk = self.chunks[current_chunk]
            # How much can we take from this chunk?
            available = chunk.length - current_index
            take = min(remaining, available)

            # Extract the slice
            part = chunk.slice(current_index, take)
            parts.append(part)
            remaining -= take

            if remaining > 0:
                # Need more data from successor chunk
                current_chunk = chunk.succ
                current_index = 0

        if parts:
            # Combine parts into sequences
            seq = {k: [p[k] for p in parts] for k in keys or parts[0].keys()}
            if concat:
                seq = {k: np.concatenate(v, 0) for k, v in seq.items()}
            return seq
        else:
            return {}

    def _setseq(self, chunkid: UUID, index: int, values: Dict[str, np.ndarray]) -> None:
        """
        Update a sequence in the buffer with new values.

        Like _getseq, this may span multiple chunks.

        Args:
            chunkid: UUID of the chunk containing the sequence start.
            index: Starting index within the chunk.
            values: Dictionary of new values to set.
        """
        remaining = self.length
        current_chunk = chunkid
        current_index = index
        value_index = 0

        while remaining > 0:
            chunk = self.chunks[current_chunk]
            available = chunk.length - current_index
            take = min(remaining, available)

            # Update this chunk's portion
            chunk_values = {k: v[value_index : value_index + take] for k, v in values.items()}
            chunk.update(current_index, take, chunk_values)

            remaining -= take
            value_index += take

            if remaining > 0:
                current_chunk = chunk.succ
                current_index = 0

    def _assemble_batch(
        self,
        seqs: List[Dict[str, List[np.ndarray]]],
        start: int,
        stop: int,
    ) -> Dict[str, np.ndarray]:
        """
        Assemble individual sequences into a batched format.

        Args:
            seqs: List of sequence dictionaries from _sample calls.
            start: Starting timestep within sequences.
            stop: Ending timestep within sequences.

        Returns:
            Dictionary mapping keys to batched arrays
        """
        if not seqs:
            return {}

        # Get all keys from the first sequence
        keys = list(seqs[0].keys())
        data = {}

        for key in keys:
            # Collect all parts for this key across sequences
            all_parts = []
            for seq in seqs:
                seq_parts = seq[key]
                # Concatenate parts within this sequence
                seq_array = np.concatenate(seq_parts, axis=0)
                # Extract the requested slice
                seq_slice = seq_array[start:stop]
                all_parts.append(seq_slice)

            # Stack all sequences for this key
            data[key] = np.stack(all_parts, axis=1)

        return data

    def _annotate_batch(
        self,
        data: Dict[str, np.ndarray],
        is_online,
        is_first: bool,
    ) -> Dict[str, np.ndarray]:
        """
        Annotate batch with metadata flags.

        Ensures 'is_first' flags are set correctly at sequence starts and that
        'is_last' flags mark episode terminations (including abandoned episodes).

        Args:
            data: Batch dictionary to annotate.
            is_online: List/tuple of boolean flags indicating which sequences are from online queue.
            is_first: Whether to mark the first timestep of each sequence as is_first=True.

        Returns:
            Annotated batch dictionary.
        """
        data = data.copy()

        if "is_first" in data:
            if is_first:
                # Mark first timestep of each sequence as is_first
                data["is_first"] = data["is_first"].copy()
                data["is_first"][0] = True

            if "is_last" in data:
                # Make sure that abandoned episodes have is_last set
                # (episodes where next step starts a new episode)
                next_is_first = np.roll(data["is_first"], shift=-1, axis=0)
                next_is_first[-1] = False
                data["is_last"] = data["is_last"] | next_is_first

        return data

    def save(self, wait: bool = False) -> None:
        """
        Save all chunks to disk asynchronously.

        Args:
            wait: If True, wait for all save operations to complete before returning.
        """
        if self.directory:
            with self.rwlock.writing:
                # Complete all active chunks before saving
                for worker, (chunkid, _) in self.current.items():
                    chunk = self.chunks[chunkid]
                    if chunk.length > 0:
                        self._complete(chunk, worker)

                # Save all unsaved chunks
                futures = []
                for chunk in self.chunks.values():
                    if not chunk.saved and chunk.uuid not in self.saved:
                        future = self.workers.submit(chunk.save, self.directory, True)
                        futures.append(future)
                        self.saved.add(chunk.uuid)

                if wait or self.save_wait:
                    # Wait for all saves to complete
                    for future in futures:
                        future.result()

    def load(
        self,
        directory: Optional[Union[str, Path]] = None,
        amount: Optional[Union[int, float]] = None,
    ) -> None:
        """
        Load chunks from disk into the buffer.

        Args:
            directory: Directory to load from. Uses self.directory if None.
            amount: Number of sequences to load. If float, treated as fraction
                   of available sequences. If None, loads all available.
        """
        directory = directory or self.directory
        if not directory:
            return

        def list_chunks(path):
            if hasattr(path, "glob"):
                return list(path.glob("*.npz"))
            else:
                return []

        directory = Path(directory)

        # Get lists of chunks already loaded and available on disk
        names_loaded = [chunk.filename for chunk in self.chunks.values()]
        names_ondisk = [x.name for x in list_chunks(directory)]
        names_ondisk = [x for x in names_ondisk if x not in names_loaded]

        if not names_ondisk:
            return

        # Determine how many chunks to load to reach target amount
        numitems = self._numitems(names_loaded + names_ondisk)
        uuids = [UUID(x.split("-")[1]) for x in names_ondisk]
        total = 0
        numchunks = 0
        for uuid in uuids:
            numchunks += 1
            total += numitems[uuid]
            if amount and total >= amount:
                break

        # Load the selected chunks
        chunks = []
        for filename in names_ondisk[:numchunks]:
            chunk = Chunk.load(directory / filename)
            if chunk:
                chunks.append(chunk)

        # Add loaded chunks to the buffer
        with self.rwlock.writing:
            self.saved.update([chunk.uuid for chunk in chunks])
            with self.refs_lock:
                # First pass: register all chunks
                for chunk in chunks:
                    self.chunks[chunk.uuid] = chunk
                    self.refs[chunk.uuid] = 0

                # Second pass: insert sequences and update reference counts
                for chunk in reversed(chunks):
                    amount = numitems[chunk.uuid]
                    self.refs[chunk.uuid] += amount
                    if chunk.succ in self.refs:
                        self.refs[chunk.succ] += 1
                    for index in range(amount):
                        self._insert(chunk.uuid, index)

    def _complete(self, chunk: Chunk, worker: int) -> Chunk:
        """
        Complete a full chunk and create its successor for continued writing.

        Called when a chunk reaches its maximum size (chunksize). Creates a new
        successor chunk and links it to the completed chunk.

        Args:
            chunk: The chunk that has reached capacity.
            worker: Worker ID that was writing to this chunk.

        Returns:
            The newly created successor chunk.
        """
        # Save chunk to disk if persistence is enabled
        if self.directory and chunk.uuid not in self.saved:
            self.workers.submit(chunk.save, self.directory, False)
            self.saved.add(chunk.uuid)

        # Create successor chunk
        succ = Chunk(self.chunksize)
        with self.refs_lock:
            self.refs[chunk.uuid] -= 1
            self.refs[succ.uuid] = 2
        self.chunks[succ.uuid] = succ
        self.current[worker] = (succ.uuid, 0)
        chunk.succ = succ.uuid  # Link to successor
        return succ

    def _numitems(self, chunks: List[Any]) -> Dict[UUID, int]:
        """
        Calculate the number of valid sequences that can start in each chunk.

        This computation accounts for:
        - Sequence length requirements (need at least `self.length` steps)
        - Chunk boundaries and successor chains
        - Future available steps in successor chunks

        Args:
            chunks: List of chunks (can be Chunk objects or filenames).

        Returns:
            Dictionary mapping chunk UUIDs to the number of sequences that can
            start in that chunk. Returns empty dict if no chunks provided.
        """
        # Convert chunks to filenames if needed
        chunks = [x.filename if hasattr(x, "filename") else x for x in chunks]
        if not chunks:
            return {}

        # Parse chunk metadata from filenames (format:
        # time-uuid-succ-length.npz)
        chunks = list(reversed(sorted([Path(x).stem for x in chunks])))
        times, uuids, succs, lengths = zip(*[x.split("-") for x in chunks])
        uuids = [UUID(x) for x in uuids]
        succs = [UUID(x) for x in succs]
        lengths = {k: int(v) for k, v in zip(uuids, lengths)}

        # Compute future available steps (steps in this chunk + all successors)
        future = {}
        for uuid, succ in zip(uuids, succs):
            future[uuid] = lengths[uuid] + future.get(succ, 0)

        # Compute number of valid sequence starts per chunk
        # A position can start a sequence if there are at least self.length
        # steps available (including future chunks in the chain)
        numitems = {}
        for uuid, succ in zip(uuids, succs):
            numitems[uuid] = lengths[uuid] + 1 - self.length + future.get(succ, 0)

        # Clip to valid range [0, chunk_length]
        numitems = {k: np.clip(v, 0, lengths[k]) for k, v in numitems.items()}
        return numitems

    def _notempty(self, reason: bool = False) -> bool:
        """
        Check if the buffer is not empty.

        Args:
            reason: Unused parameter for compatibility.

        Returns:
            True if the buffer contains sequences for sampling.
        """
        return bool(len(self.sampler))
