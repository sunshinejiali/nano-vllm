from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


# Block 缓存块，每个缓存块存储一个序列的 token _ids
# nVLLM 中实现块管理器的核心组件，负责高效管理 KV 缓存块。
# 支持前缀缓存和高效的内存复用。
class Block:

    def __init__(self, block_id):
        # block_id 缓存块 ID
        self.block_id = block_id
        # ref_count 引用计数，用于跟踪缓存块是否被其他序列引用
        self.ref_count = 0
        # hash 缓存块的哈希值，用于快速查找
        # 初始值为 -1，未被初始化
        self.hash = -1
        # token_ids 缓存块中的 token _ids 列表
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


# BlockManager 缓存块管理器，负责管理缓存块的分配、释放和查找
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        # 每个块的大小，即每个块可以存储的 token 数量
        self.block_size = block_size
        # blocks 缓存块列表，每个元素为一个缓存块
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash_to_block_id 哈希值到缓存块 ID 的映射，用于快速查找
        self.hash_to_block_id: dict[int, int] = dict()
        # free_block_ids 空闲缓存块 ID 队列，用于分配新的缓存块
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used_block_ids 已用缓存块 ID 集合，用于跟踪已分配的缓存块
        self.used_block_ids: set[int] = set()

    # compute_hash 计算缓存块的哈希值
    # 支持前缀缓存，即可以根据前一个缓存块的哈希值计算当前缓存块的哈希值
    # 前缀缓存可以避免重复计算哈希值，提高缓存利用率
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        # 如果有前缀缓存，先更新哈希值
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        # 更新哈希值，使用 token_ids 数组的字节表示
        # 确保哈希值的唯一性，即使 token_ids 相同，前缀不同也会得到不同的哈希值
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        # 分配缓存块，将其引用计数设为 1
        # 并将其从空闲队列中移除，加入已用队列
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        # 释放缓存块，将其引用计数减 1
        # 如果引用计数为 0，则将其加入空闲队列
        block = self.blocks[block_id]
        block.ref_count -= 1
        # assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        # 判断是否有足够的空闲缓存块分配给序列
        # 即序列需要的缓存块数量是否小于等于空闲缓存块数量
        return len(self.free_block_ids) >= seq.num_blocks

    # allocate 分配缓存块给序列
    # 支持前缀缓存，即可以根据前一个缓存块的哈希值计算当前缓存块的哈希值
    # 如果缓存命中，直接使用已有的缓存块，否则分配新的缓存块
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        # 遍历序列的每个缓存块
        # 计算缓存块的哈希值，根据哈希值查找缓存块
        # 如果缓存命中，直接使用已有的缓存块，否则分配新的缓存块
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 如果哈希值为 -1，说明缓存块为空，直接分配新的缓存块
            # 否则根据哈希值查找缓存块
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 如果缓存命中，增加缓存块的引用计数
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    # deallocate 释放序列使用的缓存块
    # 减少缓存块的引用计数，当引用计数为 0 时，释放缓存块
    def deallocate(self, seq: Sequence):
        # 反向遍历块表，减少引用计数，无引用时释放块。
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        # 如果序列长度为 block_size 的整数倍，说明序列最后一个缓存块已满
        # 直接分配新的缓存块，更新哈希值
        if len(seq) % self.block_size == 1:
            # 需要新块，更新哈希值
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 块已满，计算哈希
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 块未满，不计算哈希值
            assert last_block.hash == -1
