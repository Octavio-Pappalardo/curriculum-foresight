CRAFTAX_DETAILED_ENVIRONMENT_CONTEXT = """\
Craftax is a survival, exploration, crafting, and combat environment set in a procedurally generated world. The player begins empty-handed in the Overworld and can explore a sequence of regions below it. Play can involve navigating the world, maintaining the player's basic needs, gathering materials, making and improving equipment, fighting creatures, and using systems such as ranged combat, magic, and enchanting.

### The player and the world

Each episode creates a new world containing nine regions, or floors. Every floor is a 48-by-48 grid, but its terrain, resources, ladders, rooms, chests, and creature activity vary from one world to the next. The player begins at the centre of the Overworld, facing north, with no resources or equipment. Health, food, drink, energy, and mana all begin at 9; dexterity, strength, and intelligence begin at 1.

The player explores from a local viewpoint rather than receiving a complete map. It can perceive a 9-by-11 area centred on its current position, together with its inventory, equipment, survival values, experience and attributes, facing direction, current floor, learned spells, and relevant floor or encounter state. Unexplored parts of the world are not revealed through a global map. Several underground regions are dark away from their lit arrival areas, lava, and player-placed torches, further limiting what can be seen there.

Moving between floors does not reset the player or the world. Inventory, equipment, survival state, attributes, spells, opened chests, planted crops, creatures, and progress through each floor remain part of the same episode. Death, completion of the Necromancer encounter, or reaching the configured time limit ends the episode. A new episode replaces the entire world and returns the player to the initial state.

The Overworld has a day-night cycle. Night makes hostile melee creatures appear more frequently there. The underground regions have their own fixed patterns of light and darkness rather than following the surface cycle.

### Movement, facing, and actions

The player can move one tile north, south, east, or west. A movement action also changes the direction it faces, even if an obstacle prevents it from entering the destination tile. Grass, sand, paths, gravel, fire grass, and ice grass are traversable. Water, lava, rock, trees, ore deposits, walls, crops, workstations, chests, fountains, graves, and the Necromancer are not. Creatures also block the tiles they occupy.

Most interaction with the world is local. Direct interaction, placement, shooting, and spell casting operate in the direction the player is facing. Crafting is the main exception: it can use a required crafting table or furnace on any of the eight surrounding tiles, including diagonals.

The action set contains 43 actions. Choosing an action does not guarantee that its named effect occurs: an action succeeds only when all of its mechanical conditions are satisfied. The main action groups are:

| Action | Conditions and effect |
|---|---|
| No-op | Takes no deliberate action while time and the rest of the world continue to advance. |
| Move | Enters the adjacent tile only if that tile is traversable and unoccupied. The player faces the chosen direction whether or not movement succeeds. |
| Direct interaction | Acts on the adjacent tile in front of the player. A creature is attacked first if one occupies the tile. Otherwise the same action may gather or mine a block, open a chest, drink, eat a ripe crop, remove a workstation, or interact with a vulnerable Necromancer, depending on what is present and whether any tool requirement is met. |
| Place | Attempts to place a crafting table, furnace, stone block, sapling, or torch on the tile in front. It succeeds only if the player has the required material or item and the destination is valid and unoccupied. |
| Craft | Produces the chosen item only if the player has its ingredients and all required workstations are on surrounding tiles. |
| Ascend or descend | Changes floor only while the player is standing on the corresponding ladder and that direction of travel is available. |
| Shoot | Fires only if the player owns a bow, carries at least one arrow, and fewer than three player projectiles are active on the current floor. A successful shot consumes one arrow. |
| Cast a spell | Fires only if the spell has been learned, at least 2 mana is available, and fewer than three player projectiles are active on the current floor. A successful cast consumes 2 mana. |
| Use an item | Drinking a potion or reading a book requires the corresponding item and consumes it. Each potion colour has its own drinking action. |
| Sleep or rest | Begins only when the relevant recovery condition is unmet. Once begun, it temporarily overrides other chosen actions until the player has recovered or the state is interrupted. |
| Improve an attribute | Requires one unspent experience point and an attribute below its maximum level. |
| Enchant equipment | Requires the relevant equipment, the correct enchantment table on the tile directly in front of the player, the corresponding gem, and sufficient mana. |

### Regions and travel

The nine regions form one descending sequence. Each has a recognisable environment, set of resources, and family of creatures, while the detailed layout is generated anew for every episode.

| Floor | Region | Environment and notable features | Creatures |
|---:|---|---|---|
| 0 | Overworld | An outdoor landscape of grass, trees, sand, water, stone, and lava. Coal and iron occur here, while diamond is exceptionally rare. | Cow; Zombie; Skeleton |
| 1 | Dungeon | Illuminated rooms and corridors containing chests and fountains. | Snail; Orc Soldier; Orc Mage |
| 2 | Gnomish Mines | Dark caves with stone, stalagmites, water, lava, and all five ores. | Bat; Gnome Warrior; Gnome Archer |
| 3 | Sewers | Illuminated rooms and corridors containing chests, water, and an ice-enchantment table. | Snail; Lizard; Kobold |
| 4 | Vault | Illuminated rooms and corridors containing chests, fountains, and a fire-enchantment table. | Snail; Knight; Archer |
| 5 | Troll Mines | Dark caves rich in coal, iron, diamond, sapphire, and ruby, with stone, stalagmites, water, and lava. | Bat; Troll; Deep Thing |
| 6 | Fire Realm | A bright volcanic region of fire grass, fire trees, extensive lava, stone, coal, and ruby. | Bat; Pigman; Fire Elemental |
| 7 | Ice Realm | A dark frozen region of ice grass, ice shrubs, water, stone, diamond, and sapphire. | No passive creature; Frost Troll; Ice Elemental |
| 8 | Graveyard | A dark, enclosed landscape of paths, walls, and graves built around the Necromancer encounter. | Encounter waves drawn from the earlier regions |

The Dungeon, Sewers, and Vault contain fixed torches in room corners and occasional mossy sections among the walls bordering their accessible spaces. The Graveyard can also contain mossy walls. Fixed torches are part of the generated environment and cannot be collected into the inventory; mossy walls are impassable like ordinary walls.

On each floor from the Dungeon through the Ice Realm, the route down remains locked until the player has defeated eight hostile creatures on that floor. The Overworld's downward ladder does not have this kill requirement. Both melee and ranged enemies count; passive creatures do not. The count belongs to the floor and persists if the player leaves and returns. Routes upward have no kill requirement.

Using a ladder is deliberate: the player must stand on it and choose ascend or descend. Descending places the player at the upward ladder of the next region, and ascending returns it to the downward ladder of the preceding region. The Graveyard is the final floor and has no further destination below it.

### Gathering and changing the terrain

The player gathers from the tile directly in front of it. Ordinary trees, fire trees, and ice shrubs can be cut without a tool and yield one wood. Interacting with grass has a 10% chance of finding a sapling without removing the grass. Stone, stalagmites, coal, and the metal or gem deposits require progressively stronger pickaxes:

| Block | Minimum pickaxe | Yield |
|---|---|---:|
| Stone or stalagmite | Wood | 1 stone |
| Coal | Wood | 1 coal |
| Iron | Stone | 1 iron |
| Diamond | Iron | 1 diamond |
| Sapphire | Diamond | 1 sapphire |
| Ruby | Diamond | 1 ruby |

A successful mining action removes the block and adds its resource to the inventory. Using an inadequate pickaxe leaves the block unchanged.

Placement allows the player to alter its immediate surroundings. A crafting table costs 2 wood and a furnace costs 1 stone. These workstations can be placed directly without another workstation. They can later be removed by direct interaction, but removing them does not return their materials. A carried torch can be placed on grass, sand, a path, fire grass, or ice grass. It remains in the world, permanently lights a small area for the rest of the episode, and cannot be recovered into the inventory.

One stone can be placed on eligible ground or used to replace a water or lava tile. The result is a solid stone block, not a traversable path; it must be mined before the player can occupy that tile.

A sapling can be planted on an empty grass tile. It begins as an unripe plant and ripens after 600 environment steps. Eating the ripe plant restores 4 food and returns it to the unripe state, allowing the same plant to grow again.

### Crafting and ordinary equipment

Crafting uses carried materials together with nearby workstations. A crafting table or furnace counts as nearby when it occupies any of the eight tiles around the player. The recipes are:

| Item produced | Materials consumed | Required nearby workstation |
|---|---|---|
| Wood pickaxe | 1 wood | Crafting table |
| Stone pickaxe | 1 wood, 1 stone | Crafting table |
| Iron pickaxe | 1 wood, 1 stone, 1 iron, 1 coal | Crafting table and furnace |
| Diamond pickaxe | 1 wood, 3 diamond | Crafting table |
| Wood sword | 1 wood | Crafting table |
| Stone sword | 1 wood, 1 stone | Crafting table |
| Iron sword | 1 wood, 1 stone, 1 iron, 1 coal | Crafting table and furnace |
| Diamond sword | 1 wood, 2 diamond | Crafting table |
| One iron armour piece | 3 iron, 3 coal | Crafting table and furnace |
| One diamond armour piece | 3 diamond | Crafting table |
| 2 arrows | 1 wood, 1 stone | Crafting table |
| 4 torches | 1 wood, 1 coal | Crafting table |

The player carries one pickaxe and one sword, each represented by its highest acquired tier. Crafting or finding a lower-tier version does not replace better equipment. Tools, weapons, and armour do not wear out.

Armour has four slots, handled in the order helmet, chestplate, leggings, and boots. Crafting iron armour fills the first empty slot in this order. Crafting diamond armour fills or upgrades the first slot that is not already diamond. Every iron piece reduces incoming physical damage by 10%, and every diamond piece reduces it by 20%, so complete iron and diamond sets provide 40% and 80% physical defence respectively.

A bow is not craftable. It is obtained from the first chest the player opens in the Dungeon. Arrows can be crafted at any time or found in chests, but they cannot be fired until the bow has been obtained.

### Survival and recovery

Health, food, drink, and energy link survival to the passage of time. Outside the Graveyard, food, drink, and energy are gradually depleted. Health slowly recovers while food and drink are above zero and the player either has energy or is sleeping; otherwise it gradually declines. In the Graveyard, survival needs do not decrease and empty needs do not cause health loss. Reaching zero health ends the episode.

Direct interaction with water or a fountain restores up to 1 drink and resets thirst accumulation without consuming the source. A ripe plant restores up to 4 food. Defeating a passive creature with a direct attack restores up to 6 food; killing it with a projectile does not.

Sleeping can begin while energy is below its maximum. The player then remains asleep until energy is full or an attack wakes it. Sleep restores energy, speeds health and mana recovery, and slows hunger and thirst, but a sleeping player takes substantially more damage from melee attacks.

Resting can begin while health is below its maximum. It lasts until health is full, food or drink reaches zero, or an attack interrupts it. Resting prevents other actions but does not accelerate recovery in the way sleep does. Mana also regenerates gradually over time without requiring an item.

The maximum survival values depend on the player's attributes:

| Quantity | Maximum value |
|---|---:|
| Health | `8 + strength` |
| Food | `7 + 2 × dexterity` |
| Drink | `7 + 2 × dexterity` |
| Energy | `7 + 2 × dexterity` |
| Mana | `6 + 3 × intelligence` |

### Creatures and combat

Every region from the Overworld through the Ice Realm has one hostile melee creature and one hostile ranged creature. Passive creatures also appear from the Overworld through the Fire Realm, but not in the Ice Realm. Creatures spawn stochastically over time on the floor the player occupies. Passive creatures wander without attacking. Melee enemies pursue a nearby player and attack from an adjacent cardinal tile; ranged enemies generally try to keep some distance while launching projectiles. Hostiles appear more frequently until the current floor's eight-kill requirement has been met. Creatures sufficiently far from the player can disappear, so the population near the player changes during exploration.

Movement depends on creature type. Cows and snails remain on land, while bats, Fire Elementals, and Ice Elementals can fly over otherwise impassable terrain. Lizards can cross both land and water, and Deep Things live in water. Projectiles travel in cardinal directions until they hit a creature, the player, or an obstructing block. They can pass over water and lava, and a hostile projectile destroys a crafting table or furnace that it hits.

A direct interaction attacks a creature on the tile in front of the player. The physical damage before strength scaling is determined by the current sword:

| Weapon | Base physical damage |
|---|---:|
| No sword | 1 |
| Wood sword | 2 |
| Stone sword | 3 |
| Iron sword | 5 |
| Diamond sword | 8 |

A bow shot has 5 base physical damage before dexterity scaling. Shooting consumes one arrow and creates a projectile travelling in the direction the player faces.

Damage has physical, fire, and ice components. Armour and creature defences reduce components separately rather than reducing all attacks by one common amount. The ordinary creatures have the following health, attacks, and innate defences:

| Region | Passive creature | Melee enemy | Ranged enemy | Enemy defences |
|---|---|---|---|---|
| Overworld | Cow: 3 health | Zombie: 5 health; 2 physical | Skeleton: 3 health; 2 physical | None |
| Dungeon | Snail: 6 health | Orc Soldier: 9 health; 3 physical | Orc Mage: 6 health; 3 fire | None |
| Gnomish Mines | Bat: 4 health | Gnome Warrior: 7 health; 4 physical | Gnome Archer: 5 health; 2 physical | None |
| Sewers | Snail: 6 health | Lizard: 11 health; 5 physical | Kobold: 8 health; 4 physical | None |
| Vault | Snail: 6 health | Knight: 12 health; 6 physical | Archer: 12 health; 5 physical | Knight and Archer: 50% physical defence |
| Troll Mines | Bat: 4 health | Troll: 20 health; 6 physical, 1 fire, 1 ice | Deep Thing: 4 health; 4 physical, 3 fire, 3 ice | Troll: 20% physical defence; Deep Thing: none |
| Fire Realm | Bat: 4 health | Pigman: 20 health; 3 physical, 5 fire | Fire Elemental: 14 health; 3 physical, 5 fire | Pigman and Fire Elemental: 90% physical defence and complete fire defence |
| Ice Realm | None | Frost Troll: 24 health; 4 physical, 5 ice | Ice Elemental: 16 health; 4 physical, 5 ice | Frost Troll and Ice Elemental: 90% physical defence and complete ice defence |

Fire creatures lack defence against ice damage, while ice creatures lack defence against fire damage.

### Experience and attributes

The first entrance to each floor below the Overworld grants one experience point. Returning to a previously visited floor does not grant another, and ordinary creature kills do not grant experience. One point can be spent at any location to increase dexterity, strength, or intelligence by one. Each attribute begins at 1 and has a maximum of 5.

Attributes affect different parts of play:

- Each dexterity level above 1 adds 2 to maximum food, drink, and energy; slows the accumulation of hunger, thirst, and fatigue by 12.5% of their base rate; and increases arrow damage by 20%.
- Each strength level above 1 adds 1 to maximum health and increases physical melee damage by 25%.
- Each intelligence level above 1 adds 3 to maximum mana, increases fireball and iceball damage by 50%, increases the elemental damage of an enchanted sword by 5%, and accelerates mana recovery.

### Chests, potions, and magic

Opening a chest removes it from the world. Each ordinary chest can independently contain several categories of random loot:

| Loot category | Chance | Possible amount |
|---|---:|---|
| Torches | 60% | 4–7 torches |
| Ore | 60% | 1–3 coal, 1–2 iron, or 1 diamond, sapphire, or ruby |
| Potion | 50% | 1–2 potions of one colour |
| Arrows | 25% | 1–4 arrows |
| Tool or weapon | 20% | One wood, stone, iron, or diamond pickaxe or sword |

Coal and iron are more common than the other ores in chest loot. A looted pickaxe or sword upgrades the player's current item only if its tier is higher.

Some floors add a guaranteed discovery to the first chest opened there. The Dungeon's first chest supplies a bow. The first chest opened in the Sewers and the first opened in the Vault each supply one book.

There are six potion colours, but their meanings are shuffled whenever a new world is created. The colours are assigned one-to-one to six effects: restore 8 health, lose 3 health, restore 8 mana, lose 3 mana, restore 8 energy, or lose 3 energy. The assignment remains fixed within an episode but changes between episodes.

Books teach the two available spells: fireball and iceball. Reading a book consumes it and teaches one spell not already known, selected at random when both remain available. A fireball deals 3 base fire damage and an iceball 3 base ice damage before intelligence scaling. Either spell costs 2 mana per successful cast.

### Enchanting

The ice-enchantment table is found in the Sewers and the fire-enchantment table in the Vault. To enchant an item, the player must stand next to and face the appropriate table. An ice enchantment consumes 1 sapphire and a fire enchantment consumes 1 ruby; either also consumes 9 mana.

An enchanted sword adds elemental damage equal to half of the sword's base physical damage. Strength scales the physical part of the attack, while intelligence scales its elemental part. An enchanted bow similarly adds an elemental component equal to half of an arrow's base physical damage, after which dexterity scales the shot.

Armour enchantment is recorded separately for each of the four armour slots. A fire-enchanted slot reduces incoming fire damage by 20%, and an ice-enchanted slot reduces incoming ice damage by 20%. The action requires the player to own at least one armour piece, but the affected slot is selected randomly among slots that have not yet been enchanted, including currently empty slots. An enchantment on an empty slot still provides its elemental defence. Once all four slots have an enchantment, a new enchantment can replace one of the opposite type.

### The Graveyard and the Necromancer

The Graveyard contains a multi-round final encounter. Its eight rounds draw on the hostile melee and ranged creature pairs introduced across the preceding floors, beginning with Zombies and Skeletons and ending with Frost Trolls and Ice Elementals. Enemy attacks deal 1.5 times their ordinary damage in the Graveyard.

During each round, the Necromancer remains invulnerable while enemies are spawning or any spawned melee or ranged enemy remains alive. Once the wave is over, the Necromancer becomes vulnerable. Directly interacting with it in that state advances the encounter and begins the next round; ordinary weapon damage does not. Eight successful vulnerable interactions complete the encounter and end the episode.

### Variation between episodes

The broad structure and rules above remain stable, but their concrete realisation changes from episode to episode. Terrain generation changes routes and distances; resources, ladders, rooms, chests, and other structures appear in different places; and creature spawning and movement remain stochastic throughout play. Chest contents, potion-colour meanings, the order in which books teach spells, and the armour slot selected for enchanting also involve randomness. Consequently, the same objective can require different navigation, resource gathering, combat, and recovery decisions in different worlds.
"""

CUSTOM_GOAL_SUITE_CONTEXT = """\
The supplied goals belong to a custom goal suite and are not limited to Craftax’s native achievements. Treat each supplied `goal` description as the authoritative definition of its success condition. The supplied `id` is only an output identifier. Infer goal meaning and relationships from the `goal` descriptions, not from the naming or structure of the IDs.
"""

PRIMITIVE_LEARNER_CONTEXT = """\
* The learner is one shared goal-conditioned reinforcement-learning policy trained across all goals.
* In each episode, a single selected goal defines the reward. Reward is 1 only on the transition where that goal succeeds and 0 otherwise. Native Craftax rewards, progress toward the selected goal, and completion of other goals or achievements provide no reward.
* Selected-goal success immediately ends the episode and resets the environment.
* Each episode is limited to 4,096 environment steps.
* The selected textual goal is provided to the policy through a frozen language encoder.
* The language encoder may provide useful semantic structure, but the policy must learn from environment interaction how goal descriptions relate to Craftax observations, dynamics, actions, and goal success. Do not assume that the policy can reason in language or has prior knowledge of Craftax mechanics.
* The same trainable policy parameters and learned representations are used for every goal.
* Environment state resets between episodes. Inventory, location, health, encountered entities, and other episode-specific state do not persist; only changes to the learned policy persist.
* The learner has no external hierarchy or library of named skills and cannot call a previously learned goal as a subroutine.
"""

SUBROUTINE_LEARNER_CONTEXT = """\
The learner is one shared goal-conditioned reinforcement-learning policy trained from scratch across all goals. At the start of each episode, it is assigned a goal to achieve. Besides choosing primitive environment actions, it can choose to call itself with a goal it has already learned to achieve with some success. A goal becomes available for calling when its estimated success rate over completed episodes assigned that goal reaches at least 1/3. This allows behavior learned while training on one goal to be used while pursuing another.

When the policy calls a goal, it temporarily chooses actions using that goal instead of the goal assigned for the episode. It uses the behavior it has learned for the called goal in the same ongoing world, with the current inventory, location and other environment state. After 32 further environment steps, it switches back to choosing actions for the episode’s assigned goal, unless the episode has ended sooner.

* Only the goal assigned at the start of the episode determines rewarded success. Reward is 1 on the transition where that goal succeeds and 0 otherwise. That success immediately ends the episode and resets the environment.
* Completing any other goal provides no reward and does not end the episode or cause a call to finish early. Native Craftax rewards and progress toward goals also provide no reward.
* Each episode is limited to 4,096 environment steps and can also end through the environment’s ordinary terminal conditions.
* Goal descriptions are provided to the policy through a frozen language encoder. The language encoder may provide useful semantic structure, but the policy must learn from environment interaction how goal descriptions relate to Craftax observations, dynamics, actions and goal success. Do not assume that the policy can reason in language or has prior knowledge of Craftax mechanics.
* Environment state resets between episodes. Inventory, location, health, encountered entities and other episode-specific state do not persist; only changes to the learned policy persist.
"""

LEARNER_CONTEXTS = {"primitive": PRIMITIVE_LEARNER_CONTEXT, "subroutine": SUBROUTINE_LEARNER_CONTEXT}
