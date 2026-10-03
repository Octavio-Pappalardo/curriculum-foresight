from __future__ import annotations

from enum import IntEnum


class InventoryItem(IntEnum):
    WOOD = 0
    STONE = 1
    COAL = 2
    IRON = 3
    DIAMOND = 4
    SAPPHIRE = 5
    RUBY = 6
    SAPLING = 7
    TORCHES = 8
    ARROWS = 9
    BOOKS = 10


class EquipmentKind(IntEnum):
    PICKAXE = 0
    SWORD = 1
    BOW = 2
    ARMOUR = 3


class ToolTier(IntEnum):
    NONE = 0
    WOOD = 1
    STONE = 2
    IRON = 3
    DIAMOND = 4


class ArmourTier(IntEnum):
    NONE = 0
    IRON = 1
    DIAMOND = 2


class ArmourSlot(IntEnum):
    ANY = -1
    HELMET = 0
    CHESTPLATE = 1
    LEGGINGS = 2
    BOOTS = 3


class RelationMode(IntEnum):
    IN_VIEW = 0
    ADJACENT_4 = 1


class AttributeKind(IntEnum):
    DEXTERITY = 0
    STRENGTH = 1
    INTELLIGENCE = 2


class EnchantmentKind(IntEnum):
    NONE = 0
    FIRE = 1
    ICE = 2


class MobGroup(IntEnum):
    PASSIVE = 0
    MELEE = 1
    RANGED = 2
