from __future__ import annotations

from curriculum_foresight.goals.craftax_ids import CraftaxBlock, CraftaxItem, MeleeMob, PassiveMob, RangedMob
from curriculum_foresight.goals.craftax_task_generators import (
    attribute_threshold,
    block_relation,
    enchantment_present,
    equipment_tier,
    floor_kill_count,
    inventory_threshold,
    item_relation,
    mob_damaged,
    mob_defeated,
    mob_relation,
    floor_reached,
)
from curriculum_foresight.goals.goal_enums import (
    ArmourSlot,
    ArmourTier,
    AttributeKind,
    EnchantmentKind,
    EquipmentKind,
    InventoryItem,
    MobGroup,
    RelationMode,
    ToolTier,
)
from curriculum_foresight.goals.types import GoalSpaceRecipe


GOAL_SPACE_ID = "craftax_256"

_INVENTORY_THRESHOLDS = {
    InventoryItem.WOOD: (1, 2, 4, 8, 16, 32, 64),
    InventoryItem.STONE: (1, 2, 4, 8, 16, 32, 64),
    InventoryItem.COAL: (1, 2, 4, 8, 12, 24, 48),
    InventoryItem.IRON: (1, 2, 3, 6, 12, 24),
    InventoryItem.DIAMOND: (1, 2, 3, 6, 12),
    InventoryItem.SAPPHIRE: (1, 2, 4, 8),
    InventoryItem.RUBY: (1, 2, 4, 8),
    InventoryItem.SAPLING: (1, 2, 4, 8),
    InventoryItem.TORCHES: (4, 8, 16, 32, 64),
    InventoryItem.ARROWS: (2, 4, 8, 16, 32),
    InventoryItem.BOOKS: (1, 2),
}

_BLOCKS_WITH_BOTH_RELATIONS = frozenset(
    {
        CraftaxBlock.COAL,
        CraftaxBlock.IRON,
        CraftaxBlock.DIAMOND,
        CraftaxBlock.LAVA,
        CraftaxBlock.WALL_MOSS,
        CraftaxBlock.SAPPHIRE,
        CraftaxBlock.RUBY,
        CraftaxBlock.CHEST,
        CraftaxBlock.FOUNTAIN,
        CraftaxBlock.ENCHANTMENT_TABLE_FIRE,
        CraftaxBlock.ENCHANTMENT_TABLE_ICE,
        CraftaxBlock.NECROMANCER,
    }
)
_BLOCKS_WITH_ADJACENCY_ONLY = frozenset(
    {
        CraftaxBlock.WATER,
        CraftaxBlock.STONE,
        CraftaxBlock.PATH,
        CraftaxBlock.SAND,
        CraftaxBlock.CRAFTING_TABLE,
        CraftaxBlock.FURNACE,
        CraftaxBlock.PLANT,
        CraftaxBlock.RIPE_PLANT,
        CraftaxBlock.WALL,
        CraftaxBlock.STALAGMITE,
        CraftaxBlock.FIRE_GRASS,
        CraftaxBlock.ICE_GRASS,
        CraftaxBlock.FIRE_TREE,
        CraftaxBlock.ICE_SHRUB,
        CraftaxBlock.GRAVE,
    }
)
_ITEM_RELATION_SPECS = (
    (CraftaxItem.TORCH, RelationMode.ADJACENT_4, None),
    *((CraftaxItem.LADDER_DOWN, relation, floor) for floor in range(8) for relation in RelationMode),
)
_FLOOR_KILL_THRESHOLDS = {
    1: (1, 2, 4, 8),
    2: (1, 2, 4, 8),
    3: (1, 2, 4, 8),
    4: (1, 2, 4, 8),
    5: (2, 4, 8),
    6: (2, 8),
    7: (2, 8),
    8: (2, 8),
}
_MOB_TYPES_BY_GROUP = {MobGroup.PASSIVE: PassiveMob, MobGroup.MELEE: MeleeMob, MobGroup.RANGED: RangedMob}


def build_recipe() -> GoalSpaceRecipe:
    inventory_tasks = tuple(
        inventory_threshold(item, threshold) for item in InventoryItem for threshold in _INVENTORY_THRESHOLDS[item]
    )
    tool_tiers = tuple(tier for tier in ToolTier if tier != ToolTier.NONE)
    armour_tiers = tuple(tier for tier in ArmourTier if tier != ArmourTier.NONE)
    concrete_armour_slots = tuple(slot for slot in ArmourSlot if slot not in (ArmourSlot.ANY, ArmourSlot.HELMET))
    equipment_tasks = (
        *(equipment_tier(EquipmentKind.PICKAXE, tier) for tier in tool_tiers),
        *(equipment_tier(EquipmentKind.SWORD, tier) for tier in tool_tiers),
        equipment_tier(EquipmentKind.BOW),
        *(equipment_tier(EquipmentKind.ARMOUR, tier) for tier in armour_tiers),
        *(equipment_tier(EquipmentKind.ARMOUR, tier, slot) for slot in concrete_armour_slots for tier in armour_tiers),
    )
    enchantment_tasks = tuple(
        enchantment_present(equipment, enchantment)
        for enchantment in EnchantmentKind
        if enchantment != EnchantmentKind.NONE
        for equipment in (EquipmentKind.SWORD, EquipmentKind.BOW, EquipmentKind.ARMOUR)
    )
    block_relation_tasks = tuple(
        block_relation(block, relation)
        for block in CraftaxBlock
        for relation in RelationMode
        if block in _BLOCKS_WITH_BOTH_RELATIONS
        or (block in _BLOCKS_WITH_ADJACENCY_ONLY and relation == RelationMode.ADJACENT_4)
    )
    item_relation_tasks = tuple(
        item_relation(item, relation, floor=floor) for item, relation, floor in _ITEM_RELATION_SPECS
    )
    mob_relation_tasks = tuple(
        mob_relation(group, mob_type, relation)
        for group in MobGroup
        for mob_type in _MOB_TYPES_BY_GROUP[group]
        for relation in RelationMode
    )
    floor_progression_tasks = tuple(floor_reached(floor) for floor in range(1, 9))
    attribute_tasks = tuple(
        attribute_threshold(attribute, threshold) for attribute in AttributeKind for threshold in range(2, 6)
    )
    mob_combat_tasks = tuple(
        task
        for group in MobGroup
        for mob_type in _MOB_TYPES_BY_GROUP[group]
        for task in (mob_damaged(group, mob_type), mob_defeated(group, mob_type))
    )
    floor_kill_tasks = tuple(
        floor_kill_count(floor, threshold) for floor in range(1, 9) for threshold in _FLOOR_KILL_THRESHOLDS[floor]
    )

    return GoalSpaceRecipe(
        goal_space_id=GOAL_SPACE_ID,
        task_specs=(
            *inventory_tasks,
            *equipment_tasks,
            *enchantment_tasks,
            *block_relation_tasks,
            *item_relation_tasks,
            *mob_relation_tasks,
            *floor_progression_tasks,
            *attribute_tasks,
            *mob_combat_tasks,
            *floor_kill_tasks,
        ),
    )
