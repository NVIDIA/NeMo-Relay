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

/// Immutable input and geometry computed once, then reused while history is reclaimed.
/// Owns only borrowed field references; it neither allocates nor pins pool blocks.
pub(super) struct DetailPlan<'a> {
    parts: [&'a str; 3],
    needed: usize,
    block_bytes: usize,
}

/// A validated borrow prevents mutation/reclamation until copying is complete.
pub(super) struct DetailView<'a> {
    pool: &'a DetailPool,
    handle: DetailHandle,
}
impl DetailView<'_> {
    pub(super) fn len(&self) -> usize {
        self.handle.len
    }

    pub(super) fn copy_into(&self, output: &mut impl fmt::Write) -> fmt::Result {
        let mut index = self.handle.first;
        while index != NO_BLOCK {
            let block = &self.pool.blocks[index];
            let offset = index * self.pool.block_bytes;
            let text = std::str::from_utf8(&self.pool.bytes[offset..offset + block.len])
                .map_err(|_| fmt::Error)?;
            output.write_str(text)?;
            index = block.next;
        }
        Ok(())
    }
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
    pub(super) fn capacity_bytes(&self) -> usize {
        self.bytes.len()
    }
    pub(super) fn metadata_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<Block>()
    }
    /// Inspect live UTF-8 content without adding recording-path bookkeeping.
    /// Free blocks have zero length, including after reclamation.
    pub(super) fn logical_bytes(&self) -> usize {
        self.blocks.iter().map(|block| block.len).sum()
    }

    /// Plan exact UTF-8 geometry against total capacity before evicting evidence.
    pub(super) fn plan_parts<'a>(&self, parts: [&'a str; 3]) -> Option<DetailPlan<'a>> {
        let needed = TextChunks::new(&parts, self.block_bytes)
            .filter(|(_, new_block)| *new_block)
            .take(self.blocks.len().saturating_add(1))
            .count();
        (needed == 0 || (self.generation != u64::MAX && needed <= self.blocks.len())).then_some(
            DetailPlan {
                parts,
                needed,
                block_bytes: self.block_bytes,
            },
        )
    }

    /// Retry a precomputed plan as blocks become free. UTF-8 requirements are
    /// not recalculated on each reclamation; only free capacity is checked.
    pub(super) fn allocate_plan(&mut self, plan: &DetailPlan<'_>) -> Option<DetailHandle> {
        if plan.block_bytes != self.block_bytes || plan.needed > self.free_blocks {
            return None;
        }
        let needed = plan.needed;
        if needed == 0 {
            return Some(DetailHandle {
                first: NO_BLOCK,
                generation: 0,
                len: 0,
            });
        }
        if needed > self.free_blocks {
            return None;
        }
        let generation = self.generation.checked_add(1)?;
        let first = self.free_head;
        let mut current = NO_BLOCK;
        let mut len = 0;
        for (piece, new_block) in TextChunks::new(&plan.parts, self.block_bytes) {
            if new_block {
                let index = self.free_head;
                self.free_head = self.blocks[index].next;
                self.free_blocks -= 1;
                self.blocks[index] = Block {
                    next: NO_BLOCK,
                    generation,
                    len: 0,
                };
                if current != NO_BLOCK {
                    self.blocks[current].next = index;
                }
                current = index;
            }
            let block = &mut self.blocks[current];
            let offset = current * self.block_bytes + block.len;
            self.bytes[offset..offset + piece.len()].copy_from_slice(piece.as_bytes());
            block.len += piece.len();
            len += piece.len();
        }
        self.generation = generation;
        Some(DetailHandle {
            first,
            generation,
            len,
        })
    }

    fn valid(&self, handle: DetailHandle) -> bool {
        self.block_count(handle).is_some()
    }

    fn block_count(&self, handle: DetailHandle) -> Option<usize> {
        if handle.first == NO_BLOCK {
            return (handle.len == 0 && handle.generation == 0).then_some(0);
        }
        let mut index = handle.first;
        let mut len = 0usize;
        let mut seen = 0usize;
        while index != NO_BLOCK {
            let block = self.blocks.get(index)?;
            if block.len == 0 || block.generation != handle.generation {
                return None;
            }
            len = len.checked_add(block.len)?;
            seen += 1;
            if seen > self.blocks.len() {
                return None;
            }
            index = block.next;
        }
        (len == handle.len).then_some(seen)
    }

    /// Validated exclusive chain geometry; no text copy or recording ledger.
    pub(super) fn footprint(&self, handle: DetailHandle) -> Option<(usize, usize, usize)> {
        let blocks = self.block_count(handle)?;
        Some((
            blocks.checked_mul(self.block_bytes)?,
            blocks.checked_mul(size_of::<Block>())?,
            handle.len,
        ))
    }

    pub(super) fn view(&self, handle: DetailHandle) -> Option<DetailView<'_>> {
        self.valid(handle)
            .then_some(DetailView { pool: self, handle })
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
}

/// Stream complete UTF-8 fragments across field boundaries into fixed blocks.
/// A fragment starts a new block only when the previous block cannot fit its
/// next scalar. Empty fields remain represented by the event's ranges.
struct TextChunks<'parts, 'text> {
    parts: &'parts [&'text str],
    part: usize,
    offset: usize,
    used: usize,
    block_bytes: usize,
}
impl<'parts, 'text> TextChunks<'parts, 'text> {
    fn new(parts: &'parts [&'text str], block_bytes: usize) -> Self {
        Self {
            parts,
            part: 0,
            offset: 0,
            used: 0,
            block_bytes,
        }
    }
}
impl<'text> Iterator for TextChunks<'_, 'text> {
    type Item = (&'text str, bool);
    fn next(&mut self) -> Option<Self::Item> {
        while self.part < self.parts.len() {
            let remainder = &self.parts[self.part][self.offset..];
            if remainder.is_empty() {
                self.part += 1;
                self.offset = 0;
                continue;
            }
            let mut end = chunk_end(remainder, self.block_bytes - self.used);
            if end == 0 {
                self.used = 0;
                end = chunk_end(remainder, self.block_bytes);
            }
            let new_block = self.used == 0;
            self.used += end;
            self.offset += end;
            return Some((&remainder[..end], new_block));
        }
        None
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
    fn packed_pool_uses_exact_utf8_geometry_and_reuses_chains() {
        let fields = ["", "abc", "€€€", "🙂x", "xy"];
        for block_bytes in 4..=9 {
            for capacity in 0..=8 {
                let mut pool = super::DetailPool::new(capacity * block_bytes, block_bytes).unwrap();
                for (a, b, c) in fields
                    .into_iter()
                    .flat_map(|a| fields.map(|b| (a, b)))
                    .flat_map(|(a, b)| fields.map(|c| (a, b, c)))
                {
                    let parts = [a, b, c];
                    let plan = pool.plan_parts(parts);
                    let fits = plan.is_some();
                    let handle = plan.and_then(|plan| pool.allocate_plan(&plan));
                    assert_eq!(handle.is_some(), fits);
                    if let Some(handle) = handle {
                        let mut text = String::new();
                        pool.view(handle).unwrap().copy_into(&mut text).unwrap();
                        assert_eq!(text, parts.concat());
                        assert_eq!(pool.logical_bytes(), text.len());
                        pool.free(handle);
                        assert_eq!(pool.used_bytes(), 0);
                        assert_eq!(pool.logical_bytes(), 0);
                        if !text.is_empty() {
                            assert!(pool.view(handle).is_none());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn detail_plan_survives_reclamation_and_views_reject_reused_handles() {
        let mut pool = super::DetailPool::new(16, 4).unwrap();
        let old_plan = pool.plan_parts(["🙂🙂", "", ""]).unwrap();
        let first = pool.allocate_plan(&old_plan).unwrap();
        let second = pool.allocate_plan(&old_plan).unwrap();
        let plan = pool.plan_parts(["€abc", "🙂", "id"]).unwrap();
        assert!(pool.allocate_plan(&plan).is_none());
        assert!(pool.free(first));
        assert!(pool.allocate_plan(&plan).is_none());
        assert!(pool.free(second));
        let replacement = pool.allocate_plan(&plan).unwrap();
        assert!(pool.view(first).is_none());
        assert!(pool.view(second).is_none());
        assert!(!pool.free(first));
        let view = pool.view(replacement).unwrap();
        assert_eq!(view.len(), 12);
        let mut copied = String::new();
        view.copy_into(&mut copied).unwrap();
        assert_eq!(copied, "€abc🙂id");
        assert!(
            super::DetailPool::new(16, 8)
                .unwrap()
                .allocate_plan(&plan)
                .is_none()
        );
    }

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
