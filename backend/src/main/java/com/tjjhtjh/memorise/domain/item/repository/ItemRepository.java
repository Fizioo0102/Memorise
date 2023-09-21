package com.tjjhtjh.memorise.domain.item.repository;

import com.tjjhtjh.memorise.domain.item.repository.entity.Item;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface ItemRepository extends JpaRepository<Item, Long> {
    Optional<Item> findByItemSeq(Long itemSeq);
}
