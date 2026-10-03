from __future__ import annotations

import jax
import jax.numpy as jnp
from craftax.craftax.constants import MAX_OBS_DIM, MOB_ACHIEVEMENT_MAP, OBS_DIM, BlockType, ItemType

from curriculum_foresight.goals.goal_enums import (
    ArmourSlot,
    AttributeKind,
    EnchantmentKind,
    EquipmentKind,
    MobGroup,
    RelationMode,
)
from curriculum_foresight.goals.types import SuccessEvaluator


_OBS_DIM_ARRAY = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)
_CARDINAL_DIRECTIONS = jnp.array(((0, -1), (0, 1), (-1, 0), (1, 0)), dtype=jnp.int32)


def compute_goal_success(pre_state, post_state, success_evaluator, condition_params) -> jax.Array:
    evaluator = jnp.asarray(success_evaluator, dtype=jnp.int32)
    condition_params = jnp.asarray(condition_params, dtype=jnp.int32)

    is_inventory = evaluator == SuccessEvaluator.INVENTORY_AT_LEAST.value
    is_block_relation = evaluator == SuccessEvaluator.BLOCK_RELATION.value
    is_item_relation = evaluator == SuccessEvaluator.ITEM_RELATION.value
    is_item_relation_on_floor = evaluator == SuccessEvaluator.ITEM_RELATION_ON_FLOOR.value
    is_any_item_relation = is_item_relation | is_item_relation_on_floor
    is_mob_relation = evaluator == SuccessEvaluator.MOB_RELATION.value
    is_equipment = evaluator == SuccessEvaluator.EQUIPMENT_TIER_AT_LEAST.value
    is_enchantment = evaluator == SuccessEvaluator.ENCHANTMENT_PRESENT.value
    is_player_level = evaluator == SuccessEvaluator.FLOOR_AT_LEAST.value
    is_attribute = evaluator == SuccessEvaluator.ATTRIBUTE_AT_LEAST.value
    is_mob_damaged = evaluator == SuccessEvaluator.MOB_DAMAGED.value
    is_mob_defeated = evaluator == SuccessEvaluator.MOB_DEFEATED.value
    is_floor_kills = evaluator == SuccessEvaluator.FLOOR_MONSTERS_KILLED_AT_LEAST.value

    inventory_success = _compute_inventory_success(post_state, condition_params, is_inventory)
    block_relation_success = _compute_block_relation_success(post_state, condition_params, is_block_relation)
    item_relation_success = _compute_item_relation_success(post_state, condition_params, is_any_item_relation)
    item_relation_floor = jnp.where(is_item_relation_on_floor, condition_params[:, 2], 0)
    item_relation_on_floor_success = item_relation_success & (post_state.player_level == item_relation_floor)
    mob_relation_success = _compute_mob_relation_success(post_state, condition_params, is_mob_relation)
    equipment_success = _compute_equipment_success(post_state, condition_params, is_equipment)
    enchantment_success = _compute_enchantment_success(post_state, condition_params, is_enchantment)
    player_level_success = _compute_player_level_success(post_state, condition_params, is_player_level)
    attribute_success = _compute_attribute_success(post_state, condition_params, is_attribute)
    mob_damaged_success = _compute_mob_combat_success(
        pre_state, post_state, condition_params, is_mob_damaged, require_defeat=False
    )
    mob_defeated_success = _compute_mob_combat_success(
        pre_state, post_state, condition_params, is_mob_defeated, require_defeat=True
    )
    floor_kills_success = _compute_floor_kill_success(post_state, condition_params, is_floor_kills)

    return (
        (is_inventory & inventory_success)
        | (is_block_relation & block_relation_success)
        | (is_item_relation & item_relation_success)
        | (is_item_relation_on_floor & item_relation_on_floor_success)
        | (is_mob_relation & mob_relation_success)
        | (is_equipment & equipment_success)
        | (is_enchantment & enchantment_success)
        | (is_player_level & player_level_success)
        | (is_attribute & attribute_success)
        | (is_mob_damaged & mob_damaged_success)
        | (is_mob_defeated & mob_defeated_success)
        | (is_floor_kills & floor_kills_success)
    )


def _compute_inventory_success(post_state, condition_params, is_inventory):
    inventory_values = jnp.stack(
        (
            post_state.inventory.wood,
            post_state.inventory.stone,
            post_state.inventory.coal,
            post_state.inventory.iron,
            post_state.inventory.diamond,
            post_state.inventory.sapphire,
            post_state.inventory.ruby,
            post_state.inventory.sapling,
            post_state.inventory.torches,
            post_state.inventory.arrows,
            post_state.inventory.books,
        ),
        axis=-1,
    )
    item_id = jnp.where(is_inventory, condition_params[:, 0], 0)
    threshold = jnp.where(is_inventory, condition_params[:, 1], 1)
    item_count = jnp.take_along_axis(inventory_values, item_id[:, None], axis=1)[:, 0]
    return item_count >= threshold


def _compute_block_relation_success(post_state, condition_params, is_block_relation):
    target_block = jnp.where(is_block_relation, condition_params[:, 0], BlockType.INVALID.value)
    relation_mode = jnp.where(is_block_relation, condition_params[:, 1], RelationMode.IN_VIEW.value)
    floor_map = _current_floor_grid(post_state.map, post_state.player_level)
    floor_light = _current_floor_grid(post_state.light_map, post_state.player_level)
    in_view_success = _grid_block_in_view(
        floor_map, floor_light, post_state.player_position, target_block, BlockType.OUT_OF_BOUNDS.value
    )
    adjacent_success = _grid_block_adjacent_4(
        floor_map, post_state.player_position, target_block, BlockType.OUT_OF_BOUNDS.value
    )
    return jnp.where(relation_mode == RelationMode.IN_VIEW.value, in_view_success, adjacent_success)


def _compute_item_relation_success(post_state, condition_params, is_item_relation):
    target_item = jnp.where(is_item_relation, condition_params[:, 0], ItemType.TORCH.value)
    relation_mode = jnp.where(is_item_relation, condition_params[:, 1], RelationMode.IN_VIEW.value)
    floor_item_map = _current_floor_grid(post_state.item_map, post_state.player_level)
    floor_light = _current_floor_grid(post_state.light_map, post_state.player_level)
    in_view_success = _grid_entity_in_view(
        floor_item_map, floor_light, post_state.player_position, target_item, ItemType.NONE.value
    )
    adjacent_success = _grid_entity_adjacent_4(
        floor_item_map, post_state.player_position, target_item, ItemType.NONE.value
    )
    return jnp.where(relation_mode == RelationMode.IN_VIEW.value, in_view_success, adjacent_success)


def _compute_mob_relation_success(post_state, condition_params, is_mob_relation):
    mob_group = jnp.where(is_mob_relation, condition_params[:, 0], MobGroup.PASSIVE.value)
    target_type = jnp.where(is_mob_relation, condition_params[:, 1], 0)
    relation_mode = jnp.where(is_mob_relation, condition_params[:, 2], RelationMode.IN_VIEW.value)
    floor_light = _current_floor_grid(post_state.light_map, post_state.player_level)
    light_view = _local_grid_view(floor_light, post_state.player_position, 0.0) > 0.05

    passive_success = _mob_relation_for_group(
        post_state.passive_mobs,
        post_state.player_level,
        post_state.player_position,
        target_type,
        relation_mode,
        light_view,
    )
    melee_success = _mob_relation_for_group(
        post_state.melee_mobs,
        post_state.player_level,
        post_state.player_position,
        target_type,
        relation_mode,
        light_view,
    )
    ranged_success = _mob_relation_for_group(
        post_state.ranged_mobs,
        post_state.player_level,
        post_state.player_position,
        target_type,
        relation_mode,
        light_view,
    )
    return _select_mob_group_result(mob_group, passive_success, melee_success, ranged_success)


def _compute_equipment_success(post_state, condition_params, is_equipment):
    equipment_kind = jnp.where(is_equipment, condition_params[:, 0], EquipmentKind.PICKAXE.value)
    tier = jnp.where(is_equipment, condition_params[:, 1], 1)
    armour_slot = jnp.where(is_equipment, condition_params[:, 2], ArmourSlot.HELMET.value)

    is_pickaxe = equipment_kind == EquipmentKind.PICKAXE.value
    is_sword = equipment_kind == EquipmentKind.SWORD.value
    is_bow = equipment_kind == EquipmentKind.BOW.value
    is_armour = equipment_kind == EquipmentKind.ARMOUR.value

    pickaxe_success = post_state.inventory.pickaxe >= tier
    sword_success = post_state.inventory.sword >= tier
    bow_success = post_state.inventory.bow >= 1

    armour = post_state.inventory.armour
    armour_any_success = jnp.max(armour, axis=-1) >= tier
    safe_armour_slot = jnp.where(armour_slot == ArmourSlot.ANY.value, ArmourSlot.HELMET.value, armour_slot)
    concrete_armour = jnp.take_along_axis(armour, safe_armour_slot[:, None], axis=1)[:, 0]
    armour_concrete_success = concrete_armour >= tier
    armour_success = jnp.where(armour_slot == ArmourSlot.ANY.value, armour_any_success, armour_concrete_success)

    return (
        (is_pickaxe & pickaxe_success)
        | (is_sword & sword_success)
        | (is_bow & bow_success)
        | (is_armour & armour_success)
    )


def _compute_enchantment_success(post_state, condition_params, is_enchantment):
    equipment_kind = jnp.where(is_enchantment, condition_params[:, 0], EquipmentKind.SWORD.value)
    enchantment = jnp.where(is_enchantment, condition_params[:, 1], EnchantmentKind.FIRE.value)
    armour_slot = jnp.where(is_enchantment, condition_params[:, 2], ArmourSlot.HELMET.value)

    sword_success = (post_state.inventory.sword > 0) & (post_state.sword_enchantment == enchantment)
    bow_success = (post_state.inventory.bow >= 1) & (post_state.bow_enchantment == enchantment)

    armour_owned = post_state.inventory.armour > 0
    armour_enchanted = post_state.armour_enchantments == enchantment[:, None]
    armour_any_success = jnp.any(armour_owned & armour_enchanted, axis=-1)
    safe_armour_slot = jnp.where(armour_slot == ArmourSlot.ANY.value, ArmourSlot.HELMET.value, armour_slot)
    concrete_owned = jnp.take_along_axis(armour_owned, safe_armour_slot[:, None], axis=1)[:, 0]
    concrete_enchantment = jnp.take_along_axis(post_state.armour_enchantments, safe_armour_slot[:, None], axis=1)[:, 0]
    armour_concrete_success = concrete_owned & (concrete_enchantment == enchantment)
    armour_success = jnp.where(armour_slot == ArmourSlot.ANY.value, armour_any_success, armour_concrete_success)

    return (
        ((equipment_kind == EquipmentKind.SWORD.value) & sword_success)
        | ((equipment_kind == EquipmentKind.BOW.value) & bow_success)
        | ((equipment_kind == EquipmentKind.ARMOUR.value) & armour_success)
    )


def _compute_player_level_success(post_state, condition_params, is_player_level):
    min_level = jnp.where(is_player_level, condition_params[:, 0], 1)
    return post_state.player_level >= min_level


def _compute_attribute_success(post_state, condition_params, is_attribute):
    attribute_kind = jnp.where(is_attribute, condition_params[:, 0], AttributeKind.DEXTERITY.value)
    threshold = jnp.where(is_attribute, condition_params[:, 1], 1)
    attribute_values = jnp.stack(
        (post_state.player_dexterity, post_state.player_strength, post_state.player_intelligence), axis=-1
    )
    attribute_value = jnp.take_along_axis(attribute_values, attribute_kind[:, None], axis=1)[:, 0]
    return attribute_value >= threshold


def _compute_mob_combat_success(pre_state, post_state, condition_params, is_mob_combat, *, require_defeat: bool):
    mob_group = jnp.where(is_mob_combat, condition_params[:, 0], MobGroup.PASSIVE.value)
    target_type = jnp.where(is_mob_combat, condition_params[:, 1], 0)
    levels = post_state.player_level

    passive_success = _mob_combat_for_group(
        pre_state.passive_mobs, post_state.passive_mobs, levels, target_type, require_defeat=require_defeat
    )
    melee_success = _mob_combat_for_group(
        pre_state.melee_mobs, post_state.melee_mobs, levels, target_type, require_defeat=require_defeat
    )
    ranged_success = _mob_combat_for_group(
        pre_state.ranged_mobs, post_state.ranged_mobs, levels, target_type, require_defeat=require_defeat
    )
    array_success = _select_mob_group_result(mob_group, passive_success, melee_success, ranged_success)

    achievement_id = MOB_ACHIEVEMENT_MAP[mob_group, target_type]
    pre_achievement = jnp.take_along_axis(pre_state.achievements, achievement_id[:, None], axis=1)[:, 0].astype(bool)
    post_achievement = jnp.take_along_axis(post_state.achievements, achievement_id[:, None], axis=1)[:, 0].astype(bool)
    achievement_success = post_achievement & ~pre_achievement
    return array_success | achievement_success


def _compute_floor_kill_success(post_state, condition_params, is_floor_kills):
    floor = jnp.where(is_floor_kills, condition_params[:, 0], 0)
    threshold = jnp.where(is_floor_kills, condition_params[:, 1], 1)
    killed = jnp.take_along_axis(post_state.monsters_killed, floor[:, None], axis=1)[:, 0]
    return killed >= threshold


def _current_floor_grid(values, levels):
    if values.ndim == 4:
        return jnp.take_along_axis(values, levels[:, None, None, None], axis=1)[:, 0]
    return values


def _local_grid_view(floor_grid, player_position, fill_value):
    padding = MAX_OBS_DIM + 2

    def _view_one(row_grid, row_position):
        padded_grid = jnp.pad(row_grid, (padding, padding), constant_values=fill_value)
        top_left = row_position - _OBS_DIM_ARRAY // 2 + padding
        return jax.lax.dynamic_slice(padded_grid, top_left, OBS_DIM)

    return jax.vmap(_view_one)(floor_grid, player_position)


def _grid_entity_in_view(floor_grid, floor_light, player_position, target_id, fill_value):
    entity_view = _local_grid_view(floor_grid, player_position, fill_value)
    light_view = _local_grid_view(floor_light, player_position, 0.0) > 0.05
    return jnp.any((entity_view == target_id[:, None, None]) & light_view, axis=(1, 2))


def _grid_entity_adjacent_4(floor_grid, player_position, target_id, fill_value):
    adjacent_values = _adjacent_4_values(floor_grid, player_position, fill_value)
    return jnp.any(adjacent_values == target_id[:, None], axis=1)


def _grid_block_in_view(floor_grid, floor_light, player_position, target_block, fill_value):
    block_view = _local_grid_view(floor_grid, player_position, fill_value)
    light_view = _local_grid_view(floor_light, player_position, 0.0) > 0.05
    return jnp.any(_block_values_match_target(block_view, target_block) & light_view, axis=(1, 2))


def _grid_block_adjacent_4(floor_grid, player_position, target_block, fill_value):
    adjacent_values = _adjacent_4_values(floor_grid, player_position, fill_value)
    return jnp.any(_block_values_match_target(adjacent_values, target_block), axis=1)


def _block_values_match_target(block_values, target_block):
    target_shape = (target_block.shape[0],) + (1,) * (block_values.ndim - 1)
    target = target_block.reshape(target_shape)
    direct_match = block_values == target
    grave_target = (target_block == BlockType.GRAVE.value).reshape(target_shape)
    grave_match = (
        (block_values == BlockType.GRAVE.value)
        | (block_values == BlockType.GRAVE2.value)
        | (block_values == BlockType.GRAVE3.value)
    )
    return jnp.where(grave_target, grave_match, direct_match)


def _adjacent_4_values(floor_grid, player_position, fill_value):
    padded_grid = jnp.pad(floor_grid, ((0, 0), (1, 1), (1, 1)), constant_values=fill_value)
    coords = player_position[:, None, :] + _CARDINAL_DIRECTIONS[None, :, :] + 1
    batch_indices = jnp.arange(floor_grid.shape[0])[:, None]
    return padded_grid[batch_indices, coords[:, :, 0], coords[:, :, 1]]


def _mob_relation_for_group(mobs, levels, player_position, target_type, relation_mode, light_view):
    positions = _current_floor_mob_positions(mobs.position, levels)
    mask = _current_floor_mob_values(mobs.mask, levels).astype(bool)
    type_id = _current_floor_mob_values(mobs.type_id, levels)
    matches = mask & (type_id == target_type[:, None])

    delta = positions - player_position[:, None, :]
    local_position = delta + _OBS_DIM_ARRAY // 2
    on_screen = jnp.logical_and(local_position >= 0, local_position < _OBS_DIM_ARRAY).all(axis=-1)
    safe_local_x = jnp.clip(local_position[:, :, 0], 0, OBS_DIM[0] - 1)
    safe_local_y = jnp.clip(local_position[:, :, 1], 0, OBS_DIM[1] - 1)
    batch_indices = jnp.arange(positions.shape[0])[:, None]
    mob_light = light_view[batch_indices, safe_local_x, safe_local_y]
    in_view_success = jnp.any(matches & on_screen & mob_light, axis=1)

    adjacent_4 = jnp.sum(jnp.abs(delta), axis=-1) == 1
    adjacent_success = jnp.any(matches & adjacent_4, axis=1)
    return jnp.where(relation_mode == RelationMode.IN_VIEW.value, in_view_success, adjacent_success)


def _mob_combat_for_group(pre_mobs, post_mobs, levels, target_type, *, require_defeat: bool):
    pre_health = _current_floor_mob_values(pre_mobs.health, levels)
    post_health = _current_floor_mob_values(post_mobs.health, levels)
    pre_mask = _current_floor_mob_values(pre_mobs.mask, levels).astype(bool)
    pre_type_id = _current_floor_mob_values(pre_mobs.type_id, levels)
    post_type_id = _current_floor_mob_values(post_mobs.type_id, levels)
    pre_live_target = pre_mask & (pre_health > 0) & (pre_type_id == target_type[:, None])
    took_damage = post_health < pre_health
    post_same_type = post_type_id == target_type[:, None]
    event = took_damage & (post_health <= 0) if require_defeat else took_damage & post_same_type
    return jnp.any(pre_live_target & event, axis=1)


def _current_floor_mob_positions(positions, levels):
    if positions.ndim == 4:
        return jnp.take_along_axis(positions, levels[:, None, None, None], axis=1)[:, 0]
    return positions


def _current_floor_mob_values(values, levels):
    if values.ndim == 3:
        return jnp.take_along_axis(values, levels[:, None, None], axis=1)[:, 0]
    return values


def _select_mob_group_result(mob_group, passive_success, melee_success, ranged_success):
    return jnp.where(
        mob_group == MobGroup.PASSIVE.value,
        passive_success,
        jnp.where(mob_group == MobGroup.MELEE.value, melee_success, ranged_success),
    )
