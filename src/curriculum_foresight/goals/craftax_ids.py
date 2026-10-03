from __future__ import annotations

from enum import IntEnum


class CraftaxBlock(IntEnum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16
    WALL = 17
    DARKNESS = 18
    WALL_MOSS = 19
    STALAGMITE = 20
    SAPPHIRE = 21
    RUBY = 22
    CHEST = 23
    FOUNTAIN = 24
    FIRE_GRASS = 25
    ICE_GRASS = 26
    GRAVEL = 27
    FIRE_TREE = 28
    ICE_SHRUB = 29
    ENCHANTMENT_TABLE_FIRE = 30
    ENCHANTMENT_TABLE_ICE = 31
    NECROMANCER = 32
    GRAVE = 33
    GRAVE2 = 34
    GRAVE3 = 35
    NECROMANCER_VULNERABLE = 36


class CraftaxItem(IntEnum):
    NONE = 0
    TORCH = 1
    LADDER_DOWN = 2
    LADDER_UP = 3
    LADDER_DOWN_BLOCKED = 4


class PassiveMob(IntEnum):
    COW = 0
    BAT = 1
    SNAIL = 2


class MeleeMob(IntEnum):
    ZOMBIE = 0
    GNOME_WARRIOR = 1
    ORC_SOLDIER = 2
    LIZARD = 3
    KNIGHT = 4
    TROLL = 5
    PIGMAN = 6
    FROST_TROLL = 7


class RangedMob(IntEnum):
    SKELETON = 0
    GNOME_ARCHER = 1
    ORC_MAGE = 2
    KOBOLD = 3
    ARCHER = 4
    DEEP_THING = 5
    FIRE_ELEMENTAL = 6
    ICE_ELEMENTAL = 7
