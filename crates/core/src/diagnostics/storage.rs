// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixed backing storage for diagnostics. None of these operations allocate after
//! construction. Pool handles remain private to core and are never wire values.

use std::collections::TryReserveError;
use std::fmt;
use std::io;

/// Runtime-sized history with fixed backing. The wrapper deliberately exposes
/// no resize operation; displaced entries remain the state's responsibility.
pub(super) struct BoundedHistory<T> {
    buffer: circular_buffer::HeapCircularBuffer<T>,
}

impl<T> BoundedHistory<T> {
    /// Validate and account for the library's aligned header and element array.
    /// The zero-capacity fixed buffer supplies the header layout without an
    /// allocation. Actual backing is measured separately after construction.
    pub(super) fn required_bytes(capacity: usize) -> Result<usize, TryReserveError> {
        let items = std::alloc::Layout::array::<T>(capacity).map_err(|_| capacity_error())?;
        std::alloc::Layout::new::<circular_buffer::FixedCircularBuffer<T, 0>>()
            .extend(items)
            .map(|(layout, _)| layout.pad_to_align().size())
            .map_err(|_| capacity_error())
    }

    /// Call after startup admission has validated the allocation layout.
    /// The library uses the global allocation-error handler on allocation failure.
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            buffer: circular_buffer::HeapCircularBuffer::with_capacity(capacity),
        }
    }

    pub(super) fn len(&self) -> usize {
        self.buffer.len()
    }
    pub(super) fn reserved_bytes(&self) -> usize {
        std::mem::size_of_val(self.buffer.as_circular_buffer())
    }

    pub(super) fn get(&self, index: usize) -> Option<&T> {
        self.buffer.get(index)
    }

    pub(super) fn pop_front(&mut self) -> Option<T> {
        self.buffer.pop_front()
    }

    /// Return the displaced oldest entry. A zero-capacity history returns the
    /// incoming entry; its owner must classify that as omitted incoming evidence.
    pub(super) fn push_back(&mut self, value: T) -> Option<T> {
        self.buffer.push_back(value)
    }

    pub(super) fn remove(&mut self, index: usize) -> Option<T> {
        self.buffer.remove(index)
    }
}

const NO_BLOCK: usize = usize::MAX;

/// A reference to one chain of UTF-8 detail blocks. Generations prevent a
/// reclaimed chain from resolving to another observation's detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DetailHandle {
    first: usize,
    generation: u64,
    len: usize,
}

struct Block {
    next: usize,
    generation: u64,
    len: usize,
}

/// Reclaimable fixed blocks. Each block contains complete UTF-8 characters,
/// allowing borrowed formatting directly from backing storage.
pub(super) struct DetailPool {
    bytes: Box<[u8]>,
    blocks: Box<[Block]>,
    block_bytes: usize,
    free_head: usize,
    free_blocks: usize,
    generation: u64,
}

impl DetailPool {
    /// Exact fixed backing bytes, including block metadata, before allocation.
    /// The pool object itself is part of its owning state's compiled layout.
    pub(super) fn required_bytes(
        capacity_bytes: usize,
        block_bytes: usize,
    ) -> Result<usize, TryReserveError> {
        if block_bytes < 4 {
            return Err(capacity_error());
        }
        let count = capacity_bytes / block_bytes;
        count
            .checked_mul(block_bytes)
            .and_then(|data| {
                count
                    .checked_mul(std::mem::size_of::<Block>())
                    .and_then(|metadata| data.checked_add(metadata))
            })
            .ok_or_else(capacity_error)
    }
    /// Capacity is rounded down to a whole block. Blocks must hold a complete
    /// four-byte UTF-8 scalar. Invalid geometry reports a capacity error.
    pub(super) fn new(capacity_bytes: usize, block_bytes: usize) -> Result<Self, TryReserveError> {
        if block_bytes < 4 {
            return Err(capacity_error());
        }
        let count = capacity_bytes / block_bytes;
        let byte_count = count.checked_mul(block_bytes).ok_or_else(capacity_error)?;
        count
            .checked_mul(std::mem::size_of::<Block>())
            .and_then(|n| n.checked_add(byte_count))
            .ok_or_else(capacity_error)?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(byte_count)?;
        bytes.resize(byte_count, 0);
        let mut blocks = Vec::new();
        blocks.try_reserve_exact(count)?;
        blocks.extend((0..count).map(|index| Block {
            next: if index + 1 < count {
                index + 1
            } else {
                NO_BLOCK
            },
            generation: 0,
            len: 0,
        }));
        Ok(Self {
            bytes: bytes.into_boxed_slice(),
            blocks: blocks.into_boxed_slice(),
            block_bytes,
            free_head: if count == 0 { NO_BLOCK } else { 0 },
            free_blocks: count,
            generation: 0,
        })
    }

    pub(super) fn reserved_bytes(&self) -> usize {
        self.bytes.len() + self.blocks.len() * std::mem::size_of::<Block>()
    }
    pub(super) fn used_bytes(&self) -> usize {
        (self.blocks.len() - self.free_blocks) * self.block_bytes
    }

    /// Whether the bounded prefix can ever fit this pool, regardless of current
    /// occupancy. Impossible details must be omitted without evicting evidence.
    pub(super) fn can_fit(&self, text: &str, max_bytes: usize) -> bool {
        let end = chunk_end(text, max_bytes);
        let mut remainder = &text[..end];
        if remainder.is_empty() {
            return true;
        }
        if self.generation == u64::MAX {
            return false;
        }
        let mut required = 0usize;
        while !remainder.is_empty() {
            required += 1;
            if required > self.blocks.len() {
                return false;
            }
            remainder = &remainder[chunk_end(remainder, self.block_bytes)..];
        }
        true
    }

    /// Copy at most max_bytes from already-safe borrowed text. None means that
    /// the caller should reclaim evidence or omit detail; backing never grows.
    /// An empty/truncated-to-empty string consumes no block.
    pub(super) fn allocate(&mut self, text: &str, max_bytes: usize) -> Option<DetailHandle> {
        let mut end = text.len().min(max_bytes);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        self.allocate_prefix(&text[..end])
    }

    fn allocate_prefix(&mut self, text: &str) -> Option<DetailHandle> {
        if text.is_empty() {
            return Some(DetailHandle {
                first: NO_BLOCK,
                generation: 0,
                len: 0,
            });
        }
        let mut needed = 0usize;
        let mut remainder = text;
        while !remainder.is_empty() {
            let end = chunk_end(remainder, self.block_bytes);
            needed = needed.checked_add(1)?;
            if needed > self.free_blocks {
                return None;
            }
            remainder = &remainder[end..];
        }
        let generation = self.generation.checked_add(1)?;
        let first = self.free_head;
        let mut previous = NO_BLOCK;
        let mut remainder = text;
        while !remainder.is_empty() {
            let end = chunk_end(remainder, self.block_bytes);
            let index = self.free_head;
            self.free_head = self.blocks[index].next;
            self.free_blocks -= 1;
            let offset = index * self.block_bytes;
            self.bytes[offset..offset + end].copy_from_slice(&remainder.as_bytes()[..end]);
            self.blocks[index] = Block {
                next: NO_BLOCK,
                generation,
                len: end,
            };
            if previous != NO_BLOCK {
                self.blocks[previous].next = index;
            }
            previous = index;
            remainder = &remainder[end..];
        }
        self.generation = generation;
        Some(DetailHandle {
            first,
            generation,
            len: text.len(),
        })
    }

    fn valid(&self, handle: DetailHandle) -> bool {
        if handle.first == NO_BLOCK {
            return handle.len == 0 && handle.generation == 0;
        }
        let mut index = handle.first;
        let mut len = 0usize;
        let mut seen = 0usize;
        while index != NO_BLOCK {
            let Some(block) = self.blocks.get(index) else {
                return false;
            };
            if block.len == 0 || block.generation != handle.generation {
                return false;
            }
            let Some(next_len) = len.checked_add(block.len) else {
                return false;
            };
            len = next_len;
            seen += 1;
            if seen > self.blocks.len() {
                return false;
            }
            index = block.next;
        }
        len == handle.len
    }

    pub(super) fn byte_len(&self, handle: DetailHandle) -> Option<usize> {
        self.valid(handle).then_some(handle.len)
    }

    pub(super) fn free(&mut self, handle: DetailHandle) -> bool {
        if !self.valid(handle) {
            return false;
        }
        let mut index = handle.first;
        while index != NO_BLOCK {
            let next = self.blocks[index].next;
            self.blocks[index].len = 0;
            self.blocks[index].next = self.free_head;
            self.free_head = index;
            self.free_blocks += 1;
            index = next;
        }
        true
    }

    pub(super) fn copy_into(
        &self,
        handle: DetailHandle,
        output: &mut impl fmt::Write,
    ) -> fmt::Result {
        if !self.valid(handle) {
            return Err(fmt::Error);
        }
        let mut index = handle.first;
        while index != NO_BLOCK {
            let block = &self.blocks[index];
            let offset = index * self.block_bytes;
            let text = std::str::from_utf8(&self.bytes[offset..offset + block.len])
                .map_err(|_| fmt::Error)?;
            output.write_str(text)?;
            index = block.next;
        }
        Ok(())
    }
}

fn chunk_end(text: &str, max_bytes: usize) -> usize {
    let mut end = text.len().min(max_bytes);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    end
}

fn capacity_error() -> TryReserveError {
    Vec::<u8>::new()
        .try_reserve_exact(usize::MAX)
        .expect_err("usize::MAX exceeds the maximum allocation layout")
}

/// A bounded serializer/formatter destination over caller-reserved bytes.
/// Exhaustion returns an error; no fallback buffer is allocated.
pub(super) struct CappedWriter<'a> {
    buffer: &'a mut [u8],
    len: usize,
}

impl<'a> CappedWriter<'a> {
    pub(super) fn new(buffer: &'a mut [u8]) -> Self {
        Self { buffer, len: 0 }
    }
    pub(super) fn len(&self) -> usize {
        self.len
    }
}

impl io::Write for CappedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.buffer.len() - self.len {
            return Err(io::ErrorKind::WriteZero.into());
        }
        self.buffer[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl fmt::Write for CappedWriter<'_> {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        io::Write::write_all(self, value.as_bytes()).map_err(|_| fmt::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::BoundedHistory;

    #[test]
    fn history_preflight_matches_library_backing_including_alignment() {
        #[repr(align(64))]
        struct Aligned;

        fn check<T>() {
            for capacity in [0, 1, 2, 256] {
                let history = BoundedHistory::<T>::new(capacity);
                assert_eq!(
                    BoundedHistory::<T>::required_bytes(capacity).unwrap(),
                    history.reserved_bytes(),
                );
            }
        }
        check::<u8>();
        check::<u64>();
        check::<Aligned>();
        check::<super::super::state::StoredObservation>();
        assert!(BoundedHistory::<u64>::required_bytes(usize::MAX).is_err());
        assert!(BoundedHistory::<u8>::required_bytes(isize::MAX as usize).is_err());
    }
}
