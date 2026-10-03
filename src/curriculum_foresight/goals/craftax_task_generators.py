from __future__ import annotations

from enum import IntEnum
from operator import index

from curriculum_foresight.goals.craftax_ids import CraftaxBlock, CraftaxItem, MeleeMob, PassiveMob, RangedMob
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
from curriculum_foresight.goals.types import SuccessEvaluator, TaskSpec

_INVENTORY_NOUNS: dict[InventoryItem, tuple[str, str]] = {
    InventoryItem.WOOD: ("wood", "wood"),
    InventoryItem.STONE: ("stone", "stone"),
    InventoryItem.COAL: ("coal", "coal"),
    InventoryItem.IRON: ("iron", "iron"),
    InventoryItem.DIAMOND: ("diamond", "diamond"),
    InventoryItem.SAPPHIRE: ("sapphire", "sapphire"),
    InventoryItem.RUBY: ("ruby", "ruby"),
    InventoryItem.SAPLING: ("sapling", "saplings"),
    InventoryItem.TORCHES: ("torch", "torches"),
    InventoryItem.ARROWS: ("arrow", "arrows"),
    InventoryItem.BOOKS: ("book", "books"),
}
_INVENTORY_TEXT_PLURAL_OVERRIDES = {
    InventoryItem.DIAMOND: "diamonds",
    InventoryItem.SAPPHIRE: "sapphires",
    InventoryItem.RUBY: "rubies",
}

_EQUIPMENT_NOUNS: dict[EquipmentKind, str] = {
    EquipmentKind.PICKAXE: "pickaxe",
    EquipmentKind.SWORD: "sword",
    EquipmentKind.BOW: "bow",
    EquipmentKind.ARMOUR: "armour",
}

_FLOOR_NAMES: dict[int, str] = {
    0: "Overworld",
    1: "Dungeon",
    2: "Gnomish Mines",
    3: "Sewers",
    4: "Vault",
    5: "Troll Mines",
    6: "Fire Realm",
    7: "Ice Realm",
    8: "Graveyard",
}

_BLOCK_TEXT_OVERRIDES: dict[CraftaxBlock, tuple[str, str]] = {
    CraftaxBlock.WALL_MOSS: ("wall_moss", "mossy wall"),
    CraftaxBlock.ENCHANTMENT_TABLE_FIRE: ("enchantment_table_fire", "fire enchantment table"),
    CraftaxBlock.ENCHANTMENT_TABLE_ICE: ("enchantment_table_ice", "ice enchantment table"),
}
_BLOCK_RELATION_GOAL_TEXT_OVERRIDES: dict[tuple[CraftaxBlock, RelationMode], str] = {
    (CraftaxBlock.CRAFTING_TABLE, RelationMode.ADJACENT_4): "Place a crafting table.",
    (CraftaxBlock.FURNACE, RelationMode.ADJACENT_4): "Place a furnace.",
    (CraftaxBlock.PLANT, RelationMode.ADJACENT_4): "Plant a sapling.",
    (CraftaxBlock.NECROMANCER, RelationMode.IN_VIEW): "See the Necromancer.",
    (CraftaxBlock.NECROMANCER, RelationMode.ADJACENT_4): "Be adjacent to the Necromancer.",
}
_ITEM_TEXT_OVERRIDES: dict[CraftaxItem, tuple[str, str]] = {CraftaxItem.LADDER_DOWN: ("ladder_down", "downward ladder")}
_INVALID_BLOCK_RELATION_TARGETS = {
    CraftaxBlock.INVALID,
    CraftaxBlock.OUT_OF_BOUNDS,
    CraftaxBlock.DARKNESS,
    CraftaxBlock.WOOD,
    CraftaxBlock.GRAVEL,
    CraftaxBlock.GRAVE2,
    CraftaxBlock.GRAVE3,
    CraftaxBlock.NECROMANCER_VULNERABLE,
}
_INVALID_ITEM_RELATION_TARGETS = {CraftaxItem.NONE, CraftaxItem.LADDER_DOWN_BLOCKED}
_UNCOUNTABLE_ENTITY_WORDS = {
    "coal",
    "diamond",
    "fire grass",
    "grass",
    "ice grass",
    "iron",
    "lava",
    "ruby",
    "sand",
    "sapphire",
    "stone",
    "water",
}
_MOB_ENUM_BY_GROUP = {MobGroup.PASSIVE: PassiveMob, MobGroup.MELEE: MeleeMob, MobGroup.RANGED: RangedMob}


def inventory_threshold(item: InventoryItem | int, threshold: int) -> TaskSpec:
    item = _as_project_enum(InventoryItem, item, "item")
    threshold = _as_positive_int(threshold, "threshold")
    max_threshold = 2 if item == InventoryItem.BOOKS else 99
    if threshold > max_threshold:
        msg = f"{item.name.lower()} inventory thresholds must be at most {max_threshold}."
        raise ValueError(msg)
    singular, plural = _INVENTORY_NOUNS[item]
    noun = singular if threshold == 1 else _INVENTORY_TEXT_PLURAL_OVERRIDES.get(item, plural)
    item_slug = item.name.lower()

    return TaskSpec(
        task_id=f"inventory.{item_slug}.ge_{threshold}",
        goal_text=f"Have at least {threshold} {noun}.",
        success_evaluator=SuccessEvaluator.INVENTORY_AT_LEAST,
        condition_params=(item.value, threshold),
    )


def equipment_tier(
    equipment: EquipmentKind | int,
    tier: ToolTier | ArmourTier | int | None = None,
    armour_slot: ArmourSlot | int = ArmourSlot.ANY,
) -> TaskSpec:
    equipment = _as_project_enum(EquipmentKind, equipment, "equipment")
    armour_slot = _as_project_enum(ArmourSlot, armour_slot, "armour_slot")
    if equipment != EquipmentKind.ARMOUR and armour_slot != ArmourSlot.ANY:
        msg = "non-armour equipment goals must use ArmourSlot.ANY."
        raise ValueError(msg)

    if equipment in (EquipmentKind.PICKAXE, EquipmentKind.SWORD):
        tool_tier = _as_tool_tier(tier)
        tool_noun = _EQUIPMENT_NOUNS[equipment]
        tier_slug = tool_tier.name.lower()
        or_better = "" if tool_tier == ToolTier.DIAMOND else " or better"
        return TaskSpec(
            task_id=f"equipment.{tool_noun}.ge_{tier_slug}",
            goal_text=f"Obtain {_article_for(tier_slug)} {tier_slug} {tool_noun}{or_better}.",
            success_evaluator=SuccessEvaluator.EQUIPMENT_TIER_AT_LEAST,
            condition_params=(equipment.value, tool_tier.value, ArmourSlot.ANY.value),
        )

    if equipment == EquipmentKind.BOW:
        if tier is not None and _as_int(tier, "tier") != 1:
            msg = "bow equipment goals only support presence with tier 1."
            raise ValueError(msg)
        return TaskSpec(
            task_id="equipment.bow.present",
            goal_text="Obtain a bow.",
            success_evaluator=SuccessEvaluator.EQUIPMENT_TIER_AT_LEAST,
            condition_params=(equipment.value, 1, ArmourSlot.ANY.value),
        )

    armour_tier = _as_armour_tier(tier)
    tier_slug = armour_tier.name.lower()
    slot_slug = armour_slot.name.lower()
    if armour_slot == ArmourSlot.ANY:
        or_better = " or better" if armour_tier == ArmourTier.IRON else ""
        goal_text = f"Obtain one piece of {tier_slug} armour{or_better}."
    else:
        article = "" if armour_slot in (ArmourSlot.LEGGINGS, ArmourSlot.BOOTS) else f"{_article_for(tier_slug)} "
        or_better = " or better" if armour_tier == ArmourTier.IRON else ""
        goal_text = f"Obtain {article}{tier_slug} {slot_slug}{or_better}."

    return TaskSpec(
        task_id=f"equipment.armour.{slot_slug}.ge_{tier_slug}",
        goal_text=goal_text,
        success_evaluator=SuccessEvaluator.EQUIPMENT_TIER_AT_LEAST,
        condition_params=(equipment.value, armour_tier.value, armour_slot.value),
    )


def block_relation(block: CraftaxBlock | int, relation_mode: RelationMode | int) -> TaskSpec:
    block = _as_project_enum(CraftaxBlock, block, "block")
    if block in _INVALID_BLOCK_RELATION_TARGETS:
        msg = "Unsupported block for a relation goal."
        raise ValueError(msg)
    relation_mode = _as_project_enum(RelationMode, relation_mode, "relation_mode")
    slug, phrase = _enum_slug_and_phrase(block, _BLOCK_TEXT_OVERRIDES)
    relation_slug = _relation_slug(relation_mode)

    goal_text = _BLOCK_RELATION_GOAL_TEXT_OVERRIDES.get((block, relation_mode))
    if goal_text is None:
        if relation_mode == RelationMode.IN_VIEW:
            goal_text = f"See {_entity_phrase(phrase)}."
        else:
            goal_text = f"Be adjacent to {_entity_phrase(phrase)}."

    return TaskSpec(
        task_id=f"relation.block.{slug}.{relation_slug}",
        goal_text=goal_text,
        success_evaluator=SuccessEvaluator.BLOCK_RELATION,
        condition_params=(block.value, relation_mode.value),
    )


def item_relation(item: CraftaxItem | int, relation_mode: RelationMode | int, *, floor: int | None = None) -> TaskSpec:
    item = _as_project_enum(CraftaxItem, item, "item")
    if item in _INVALID_ITEM_RELATION_TARGETS:
        msg = "Unsupported item for a relation goal."
        raise ValueError(msg)
    relation_mode = _as_project_enum(RelationMode, relation_mode, "relation_mode")
    slug, phrase = _enum_slug_and_phrase(item, _ITEM_TEXT_OVERRIDES)
    relation_slug = _relation_slug(relation_mode)

    if floor is None:
        if relation_mode == RelationMode.IN_VIEW:
            goal_text = f"See {_article_phrase(phrase)}."
        else:
            goal_text = f"Be adjacent to {_article_phrase(phrase)}."

        return TaskSpec(
            task_id=f"relation.item.{slug}.{relation_slug}",
            goal_text=goal_text,
            success_evaluator=SuccessEvaluator.ITEM_RELATION,
            condition_params=(item.value, relation_mode.value),
        )

    floor = _as_floor(floor, "floor")
    if item == CraftaxItem.LADDER_DOWN and floor == 8:
        msg = "downward-ladder relation goals require a floor from 0 to 7."
        raise ValueError(msg)
    if item == CraftaxItem.LADDER_UP and floor in (0, 8):
        msg = "upward-ladder relation goals require a floor from 1 to 7."
        raise ValueError(msg)

    floor_name = _FLOOR_NAMES[floor]
    floor_target = f"the {phrase}" if item == CraftaxItem.LADDER_DOWN else _article_phrase(phrase)
    if relation_mode == RelationMode.IN_VIEW:
        goal_text = f"See {floor_target} on floor {floor}, the {floor_name}."
    else:
        goal_text = f"Be adjacent to {floor_target} on floor {floor}, the {floor_name}."

    return TaskSpec(
        task_id=f"relation.item.{slug}.floor_{floor}.{relation_slug}",
        goal_text=goal_text,
        success_evaluator=SuccessEvaluator.ITEM_RELATION_ON_FLOOR,
        condition_params=(item.value, relation_mode.value, floor),
    )


def mob_relation(
    mob_group: MobGroup | int, mob_type: PassiveMob | MeleeMob | RangedMob | int, relation_mode: RelationMode | int
) -> TaskSpec:
    mob_group = _as_project_enum(MobGroup, mob_group, "mob_group")
    mob_type = _as_mob_type_for_group(mob_group, mob_type)
    relation_mode = _as_project_enum(RelationMode, relation_mode, "relation_mode")
    slug, phrase = _mob_slug_and_phrase(mob_group, mob_type)
    relation_slug = _relation_slug(relation_mode)

    if relation_mode == RelationMode.IN_VIEW:
        goal_text = f"See {_article_phrase(phrase)}."
    else:
        goal_text = f"Be adjacent to {_article_phrase(phrase)}."

    return TaskSpec(
        task_id=f"relation.mob.{mob_group.name.lower()}.{slug}.{relation_slug}",
        goal_text=goal_text,
        success_evaluator=SuccessEvaluator.MOB_RELATION,
        condition_params=(mob_group.value, mob_type.value, relation_mode.value),
    )


def enchantment_present(
    equipment: EquipmentKind | int, enchantment: EnchantmentKind | int, armour_slot: ArmourSlot | int = ArmourSlot.ANY
) -> TaskSpec:
    equipment = _as_project_enum(EquipmentKind, equipment, "equipment")
    enchantment = _as_project_enum(EnchantmentKind, enchantment, "enchantment")
    armour_slot = _as_project_enum(ArmourSlot, armour_slot, "armour_slot")

    if equipment not in (EquipmentKind.SWORD, EquipmentKind.BOW, EquipmentKind.ARMOUR):
        msg = "enchantment goals require sword, bow, or armour equipment."
        raise ValueError(msg)
    if enchantment == EnchantmentKind.NONE:
        msg = "enchantment goals require FIRE or ICE."
        raise ValueError(msg)
    if equipment != EquipmentKind.ARMOUR and armour_slot != ArmourSlot.ANY:
        msg = "non-armour enchantment goals must use ArmourSlot.ANY."
        raise ValueError(msg)

    enchantment_slug = enchantment.name.lower()
    equipment_slug = _EQUIPMENT_NOUNS[equipment]
    enchantment_text = f"{_article_for(enchantment_slug)} {enchantment_slug} enchantment"

    if equipment == EquipmentKind.ARMOUR:
        slot_slug = armour_slot.name.lower()
        if armour_slot == ArmourSlot.ANY:
            target_text = "one piece of armour"
        else:
            target_text = slot_slug if armour_slot in (ArmourSlot.LEGGINGS, ArmourSlot.BOOTS) else f"a {slot_slug}"
        goal_text = f"Apply {enchantment_text} to {target_text}."
        task_id = f"enchantment.armour.{slot_slug}.{enchantment_slug}"
    else:
        goal_text = f"Apply {enchantment_text} to a {equipment_slug}."
        task_id = f"enchantment.{equipment_slug}.{enchantment_slug}"

    return TaskSpec(
        task_id=task_id,
        goal_text=goal_text,
        success_evaluator=SuccessEvaluator.ENCHANTMENT_PRESENT,
        condition_params=(equipment.value, enchantment.value, armour_slot.value),
    )


def floor_reached(level: int) -> TaskSpec:
    level = _as_floor(level, "level")
    if level == 0:
        msg = "player level goals require a floor from 1 to 8."
        raise ValueError(msg)
    floor_name = _FLOOR_NAMES[level]
    return TaskSpec(
        task_id=f"player_level.ge_{level}",
        goal_text=f"Reach floor {level}, the {floor_name}.",
        success_evaluator=SuccessEvaluator.FLOOR_AT_LEAST,
        condition_params=(level,),
    )


def attribute_threshold(attribute: AttributeKind | int, threshold: int) -> TaskSpec:
    attribute = _as_project_enum(AttributeKind, attribute, "attribute")
    threshold = _as_int(threshold, "threshold")
    if not 2 <= threshold <= 5:
        msg = "attribute thresholds must be between 2 and 5."
        raise ValueError(msg)
    attribute_slug = attribute.name.lower()
    return TaskSpec(
        task_id=f"attribute.{attribute_slug}.ge_{threshold}",
        goal_text=f"Reach {attribute_slug} level {threshold}.",
        success_evaluator=SuccessEvaluator.ATTRIBUTE_AT_LEAST,
        condition_params=(attribute.value, threshold),
    )


def mob_damaged(mob_group: MobGroup | int, mob_type: PassiveMob | MeleeMob | RangedMob | int) -> TaskSpec:
    return _mob_combat_task(mob_group, mob_type, SuccessEvaluator.MOB_DAMAGED, "damaged", "Damage")


def mob_defeated(mob_group: MobGroup | int, mob_type: PassiveMob | MeleeMob | RangedMob | int) -> TaskSpec:
    return _mob_combat_task(mob_group, mob_type, SuccessEvaluator.MOB_DEFEATED, "defeated", "Defeat")


def floor_kill_count(floor: int, threshold: int) -> TaskSpec:
    floor = _as_floor(floor, "floor")
    if floor == 0:
        msg = "floor kill-count goals require a floor from 1 to 8."
        raise ValueError(msg)
    threshold = _as_positive_int(threshold, "threshold")
    floor_name = _FLOOR_NAMES[floor]
    monster_text = "hostile monster" if threshold == 1 else "hostile monsters"
    return TaskSpec(
        task_id=f"floor_kills.floor_{floor}.ge_{threshold}",
        goal_text=f"Kill at least {threshold} {monster_text} on floor {floor}, the {floor_name}.",
        success_evaluator=SuccessEvaluator.FLOOR_MONSTERS_KILLED_AT_LEAST,
        condition_params=(floor, threshold),
    )


def _as_tool_tier(tier: ToolTier | ArmourTier | int | None) -> ToolTier:
    if tier is None:
        msg = "Pickaxe and sword goals require a tool tier other than NONE."
        raise ValueError(msg)
    tool_tier = _as_project_enum(ToolTier, tier, "tier")
    if tool_tier == ToolTier.NONE:
        msg = "Pickaxe and sword goals require a tool tier other than NONE."
        raise ValueError(msg)
    return tool_tier


def _as_armour_tier(tier: ToolTier | ArmourTier | int | None) -> ArmourTier:
    if tier is None:
        msg = "Armour goals require an armour tier other than NONE."
        raise ValueError(msg)
    armour_tier = _as_project_enum(ArmourTier, tier, "tier")
    if armour_tier == ArmourTier.NONE:
        msg = "Armour goals require an armour tier other than NONE."
        raise ValueError(msg)
    return armour_tier


def _as_positive_int(value: int, field_name: str) -> int:
    int_value = _as_int(value, field_name)
    if int_value <= 0:
        msg = f"{field_name} must be positive."
        raise ValueError(msg)
    return int_value


def _as_floor(value: int, field_name: str) -> int:
    floor = _as_int(value, field_name)
    if floor not in _FLOOR_NAMES:
        msg = f"{field_name} must be a Craftax floor from 0 to 8."
        raise ValueError(msg)
    return floor


def _as_project_enum(enum_type, value, field_name: str):
    if isinstance(value, enum_type):
        return value
    if isinstance(value, (bool, IntEnum)):
        msg = f"{field_name} enum must be {enum_type.__name__}, got {type(value).__name__}."
        raise ValueError(msg)
    try:
        raw_value = index(value)
    except TypeError as exc:
        msg = f"{field_name} must be {enum_type.__name__} or an integer."
        raise ValueError(msg) from exc
    return enum_type(raw_value)


def _as_int(value: int, field_name: str) -> int:
    if isinstance(value, (bool, IntEnum)):
        msg = f"{field_name} must be an integer, not {type(value).__name__}."
        raise ValueError(msg)
    try:
        int_value = index(value)
    except TypeError as exc:
        msg = f"{field_name} must be an integer."
        raise ValueError(msg) from exc
    return int_value


def _as_mob_type_for_group(
    mob_group: MobGroup, mob_type: PassiveMob | MeleeMob | RangedMob | int
) -> PassiveMob | MeleeMob | RangedMob:
    return _as_project_enum(_MOB_ENUM_BY_GROUP[mob_group], mob_type, f"{mob_group.name.lower()} mob_type")


def _mob_slug_and_phrase(mob_group: MobGroup, mob_type: PassiveMob | MeleeMob | RangedMob) -> tuple[str, str]:
    words = _identifier_words(mob_type.name)
    slug = "_".join(words)
    phrase = " ".join(words)
    if mob_group == MobGroup.RANGED and mob_type == RangedMob.DEEP_THING:
        phrase = "Deep Thing"
    return slug, phrase


def _identifier_words(identifier: str) -> tuple[str, ...]:
    return tuple(part for part in identifier.replace("-", " ").replace("_", " ").lower().split() if part)


def _enum_slug_and_phrase(enum_member: IntEnum, text_overrides: dict[IntEnum, tuple[str, str]]) -> tuple[str, str]:
    override = text_overrides.get(enum_member)
    if override is not None:
        return override
    words = _identifier_words(enum_member.name)
    return "_".join(words), " ".join(words)


def _relation_slug(relation_mode: RelationMode) -> str:
    if relation_mode == RelationMode.IN_VIEW:
        return "in_view"
    return "adjacent_4"


def _mob_combat_task(
    mob_group: MobGroup | int,
    mob_type: PassiveMob | MeleeMob | RangedMob | int,
    success_evaluator: SuccessEvaluator,
    task_suffix: str,
    verb: str,
) -> TaskSpec:
    mob_group = _as_project_enum(MobGroup, mob_group, "mob_group")
    mob_type = _as_mob_type_for_group(mob_group, mob_type)
    slug, phrase = _mob_slug_and_phrase(mob_group, mob_type)
    return TaskSpec(
        task_id=f"mob.{mob_group.name.lower()}.{slug}.{task_suffix}",
        goal_text=f"{verb} {_article_phrase(phrase)}.",
        success_evaluator=success_evaluator,
        condition_params=(mob_group.value, mob_type.value),
    )


def _entity_phrase(phrase: str) -> str:
    if phrase in _UNCOUNTABLE_ENTITY_WORDS:
        return phrase
    return _article_phrase(phrase)


def _article_phrase(phrase: str) -> str:
    return f"{_article_for(phrase)} {phrase}"


def _article_for(next_word: str) -> str:
    return "an" if next_word[0].lower() in "aeiou" else "a"
